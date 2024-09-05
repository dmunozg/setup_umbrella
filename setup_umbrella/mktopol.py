#!/usr/bin/env python3

import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd


def chunks(lst: list[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_cpu_count() -> int:
    """Detecta el número de núcleos disponibles, identifica si está dentro de
    una tarea SLURM"""
    slurm_cpu_count = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpu_count:
        return int(slurm_cpu_count)
    else:
        return mp.cpu_count()


def parse_atoms_in_gro(line_list: list[str]) -> pd.DataFrame:
    atomInfoParser_NullVelocity = re.compile(
        r"(?P<resNumber>[\d\ ]{5})(?P<resName>[\s\d\w]{5})(?P<atomName>[\s\d\w]{5})(?P<atomNumber>[\s\d]{5})(?P<posX>[-\s.\d]{8})(?P<posY>[-\s.\d]{8})(?P<posZ>[-\s.\d]{8})"
    )
    atomInfoParser = re.compile(
        r"(?P<resNumber>[\d\ ]{5})(?P<resName>[\s\d\w]{5})(?P<atomName>[\s\d\w]{5})(?P<atomNumber>[\s\d]{5})(?P<posX>[-\s.\d]{8})(?P<posY>[-\s.\d]{8})(?P<posZ>[-\s.\d]{8})(?P<velX>[-\s.\d]{8})(?P<velY>[-\s.\d]{8})(?P<velZ>[-\s.\d]{8})"
    )
    resultingDataframe = pd.DataFrame(
        columns=[
            "resNumber",
            "resName",
            "atomName",
            "atomNumber",
            "posX",
            "posY",
            "posZ",
            "velX",
            "velY",
            "velZ",
        ]
    )
    for line in line_list:
        if atomInfoParser.match(line):
            # Extraer datos, incluyendo velocidad
            currentLineParser = atomInfoParser.match(line)
            currentLineInfo = {
                "resNumber": int(currentLineParser["resNumber"]),
                "resName": currentLineParser["resName"].strip(),
                "atomName": currentLineParser["atomName"].strip(),
                "atomNumber": int(currentLineParser["atomNumber"]),
                "posX": float(currentLineParser["posX"]),
                "posY": float(currentLineParser["posY"]),
                "posZ": float(currentLineParser["posZ"]),
                "velX": float(currentLineParser["velX"]),
                "velY": float(currentLineParser["velY"]),
                "velZ": float(currentLineParser["velZ"]),
            }
            resultingDataframe = pd.concat(
                [
                    resultingDataframe,
                    pd.DataFrame(currentLineInfo),
                ],
                ignore_index=True,
            )
            pass
        elif atomInfoParser_NullVelocity.match(line):
            # Extraer datos, reportar velocidad como NaN
            currentLineParser = atomInfoParser_NullVelocity.match(line)
            currentLineInfo = {
                "resNumber": int(currentLineParser["resNumber"]),
                "resName": currentLineParser["resName"].strip(),
                "atomName": currentLineParser["atomName"].strip(),
                "atomNumber": int(currentLineParser["atomNumber"]),
                "posX": float(currentLineParser["posX"]),
                "posY": float(currentLineParser["posY"]),
                "posZ": float(currentLineParser["posZ"]),
                "velX": np.nan,
                "velY": np.nan,
                "velZ": np.nan,
            }
            resultingDataframe = pd.concat(
                [
                    resultingDataframe,
                    pd.DataFrame(currentLineInfo),
                ],
                ignore_index=True,
            )
            pass
        else:
            # Error de linea desconocida
            print("ERROR: Linea no reconocida:")
            print(line)
    return resultingDataframe


def gro_to_dataframe(
    gro_filename: str | Path,
) -> tuple[str, int, pd.DataFrame, tuple[float, float, float]]:
    """Returns a tuple with the following information:
    (title, nAtoms, atomDataframe, dimensions)
    Multi threaded implementation"""
    inputGroFile = open(gro_filename)
    print("Cargando archivo GRO en la memoria... ", end="")
    inputGroFileLines = inputGroFile.readlines()
    print("OK!")
    inputGroFile.close()
    # Extraer datos de título, número de átomos, y dimensiones de la caja
    print("Extrayendo metadatos de la caja... ", end="")
    title = inputGroFileLines.pop(0).rstrip()
    nAtoms = int(inputGroFileLines.pop(0))
    dimentionsDataRaw = inputGroFileLines.pop(-1)
    dimentionsData = dimentionsDataRaw.split()
    dimX_str, dimY_str, dimZ_str = (
        dimentionsData[0],
        dimentionsData[1],
        dimentionsData[2],
    )
    dimX, dimY, dimZ = float(dimX_str), float(dimY_str), float(dimZ_str)
    print("OK!")
    # Dividir la información cruda de los átomos según la cantidad de núcleos
    # disponibles
    print(
        "Importando átomos empleando {} núcleos... ".format(get_cpu_count()),
        end="",
    )
    atomsPerCPU = len(inputGroFileLines) / get_cpu_count()
    splitLines = chunks(inputGroFileLines, int(atomsPerCPU))
    # Repartir cada grupo a cada núcleo para que cada uno genere un dataFrame
    # independiente
    with mp.Pool(get_cpu_count()) as workerPool:
        splitDataFrames = workerPool.map(parse_atoms_in_gro, list(splitLines))
    print("OK!")
    # Unir los dataframes resultantes
    scrambledResultingDataFrame = pd.concat(splitDataFrames, ignore_index=True)
    # Reordenar resultados
    resultingDataframe = scrambledResultingDataFrame.sort_values(
        ["atomNumber"], ignore_index=True
    )
    return (title, nAtoms, resultingDataframe, (dimX, dimY, dimZ))


def main(gro_file: str, topol_file: Optional[str] = None) -> int:
    # Parsear el archivo .gro completo
    mdDataFrame = gro_to_dataframe(gro_file)[2]
    # Generar un DataFrame con la topología
    topologyMoleculesDataFrame = pd.DataFrame(columns=["resName", "mols"])
    # Datos del primer residuo:
    currResidueName = mdDataFrame["resName"][0]
    currResidueNumber = mdDataFrame["resNumber"][0]
    currResidueMols = 1
    print("Generando topología... ", end="")
    for _idx, row in mdDataFrame[["resName", "resNumber"]].iterrows():
        if (currResidueName == row["resName"]) & (
            currResidueNumber == row["resNumber"]
        ):
            pass
        elif (currResidueName == row["resName"]) & (
            currResidueNumber != row["resNumber"]
        ):
            currResidueMols += 1
            currResidueNumber = row["resNumber"]
            pass
        else:
            topologyMoleculesDataFrame = pd.concat(
                [
                    topologyMoleculesDataFrame,
                    pd.DataFrame({
                        "resName": currResidueName,
                        "mols": currResidueMols,
                    }),
                ],
                ignore_index=True,
            )
            currResidueName = row["resName"]
            currResidueNumber = row["resNumber"]
            currResidueMols = 1
    topologyMoleculesDataFrame = pd.concat(
        [
            topologyMoleculesDataFrame,
            pd.DataFrame({
                "resName": currResidueName,
                "mols": currResidueMols,
            }),
        ],
        ignore_index=True,
    )
    print("OK!")
    print("[ molecules ]")
    print("; Compound\t#Mols")
    for _idx, row in topologyMoleculesDataFrame.iterrows():
        print(
            "{resName}\t\t{nMols}".format(
                resName=row["resName"], nMols=row["mols"]
            )
        )
    return 0


def run() -> None:
    import argparse
    from sys import exit

    parser = argparse.ArgumentParser(
        description="Genera la entrada '[molecules]' para un archivo de topología de Gromacs"
    )
    parser.add_argument(
        "gro_file",
        metavar="Archivo.gro",
        type=str,
        help="Archivo .gro del cual se generará la topología",
    )
    args = parser.parse_args()
    exit(main(gro_file=args.gro_file))


if __name__ == "__main__":
    run()
