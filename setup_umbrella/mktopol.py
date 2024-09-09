#!/usr/bin/env python3

import multiprocessing as mp
import os
import re
from pathlib import Path
from sys import stdout
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
from loguru import logger


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
        r"(?P<resNumber>[\d\ ]{5})(?P<resName>[\s\d\w]{5})"
        r"(?P<atomName>[\s\d\w]{5})(?P<atomNumber>[\s\d]{5})"
        r"(?P<posX>[-\s.\d]{8})(?P<posY>[-\s.\d]{8})(?P<posZ>[-\s.\d]{8})"
    )
    atomInfoParser = re.compile(
        r"(?P<resNumber>[\d\ ]{5})(?P<resName>[\s\d\w]{5})"
        r"(?P<atomName>[\s\d\w]{5})(?P<atomNumber>[\s\d]{5})"
        r"(?P<posX>[-\s.\d]{8})(?P<posY>[-\s.\d]{8})(?P<posZ>[-\s.\d]{8})"
        r"(?P<velX>[-\s.\d]{8})"
        r"(?P<velY>[-\s.\d]{8})(?P<velZ>[-\s.\d]{8})"
    )
    parsed_data: list[dict[str, Any]] = []
    for line in line_list:
        parsed_line = atomInfoParser.match(line)
        parsed_line_null_vel = atomInfoParser_NullVelocity.match(line)
        if parsed_line:
            # Extraer datos, incluyendo velocidad
            current_line_info = {
                "resNumber": int(parsed_line["resNumber"]),
                "resName": parsed_line["resName"].strip(),
                "atomName": parsed_line["atomName"].strip(),
                "atomNumber": int(parsed_line["atomNumber"]),
                "posX": float(parsed_line["posX"]),
                "posY": float(parsed_line["posY"]),
                "posZ": float(parsed_line["posZ"]),
                "velX": float(parsed_line["velX"]),
                "velY": float(parsed_line["velY"]),
                "velZ": float(parsed_line["velZ"]),
            }
            parsed_data.append(current_line_info)
            pass
        elif parsed_line_null_vel:
            # Extraer datos, reportar velocidad como NaN
            current_line_info = {
                "resNumber": int(parsed_line_null_vel["resNumber"]),
                "resName": parsed_line_null_vel["resName"].strip(),
                "atomName": parsed_line_null_vel["atomName"].strip(),
                "atomNumber": int(parsed_line_null_vel["atomNumber"]),
                "posX": float(parsed_line_null_vel["posX"]),
                "posY": float(parsed_line_null_vel["posY"]),
                "posZ": float(parsed_line_null_vel["posZ"]),
                "velX": np.nan,
                "velY": np.nan,
                "velZ": np.nan,
            }
            parsed_data.append(current_line_info)
            pass
        else:
            # Error de linea desconocida
            logger.error("Linea no reconocida:")
            logger.error(line)
    return pd.DataFrame(parsed_data)


def gro_to_dataframe(
    gro_filename: str | Path,
) -> tuple[str, int, pd.DataFrame, tuple[float, float, float]]:
    """Returns a tuple with the following information:
    (title, nAtoms, atomDataframe, dimensions)
    Multi threaded implementation"""

    logger.info("Cargando archivo GRO en la memoria... ")
    with open(gro_filename) as input_gro_file:
        input_gro_lines = input_gro_file.readlines()
    # Extraer datos de título, número de átomos, y dimensiones de la caja
    logger.info("Extrayendo metadatos de la caja... ")
    title = input_gro_lines.pop(0).rstrip()
    nAtoms = int(input_gro_lines.pop(0))
    dimentionsDataRaw = input_gro_lines.pop(-1)
    dimentionsData = dimentionsDataRaw.split()
    dimX_str, dimY_str, dimZ_str = (
        dimentionsData[0],
        dimentionsData[1],
        dimentionsData[2],
    )
    dimX, dimY, dimZ = float(dimX_str), float(dimY_str), float(dimZ_str)
    # Dividir la información cruda de los átomos según la cantidad de núcleos
    # disponibles
    logger.info(
        f"Importando átomos empleando {get_cpu_count()} núcleos... ",
    )
    atomsPerCPU = len(input_gro_lines) / get_cpu_count()
    splitLines = chunks(input_gro_lines, int(atomsPerCPU))
    # Repartir cada grupo a cada núcleo para que cada uno genere un dataFrame
    # independiente
    with mp.Pool(get_cpu_count()) as workerPool:
        splitDataFrames = workerPool.map(parse_atoms_in_gro, list(splitLines))
    # Unir los dataframes resultantes
    scrambledResultingDataFrame = pd.concat(splitDataFrames, ignore_index=True)
    # Reordenar resultados
    resultingDataframe = scrambledResultingDataFrame.sort_values(
        ["atomNumber"], ignore_index=True
    )
    return (title, nAtoms, resultingDataframe, (dimX, dimY, dimZ))


@logger.catch
def main(gro_file: str, topol_file: Optional[str] = None) -> int:
    # Parsear el archivo .gro completo
    mdDataFrame = gro_to_dataframe(gro_file)[2]
    # Generar un DataFrame con la topología
    topologyMoleculesDataFrame = pd.DataFrame(columns=["resName", "mols"])
    # Datos del primer residuo:
    currResidueName = mdDataFrame["resName"][0]
    currResidueNumber = mdDataFrame["resNumber"][0]
    currResidueMols = 1
    logger.info("Generando topología... ")
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
                    pd.DataFrame(
                        {
                            "resName": currResidueName,
                            "mols": currResidueMols,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
            currResidueName = row["resName"]
            currResidueNumber = row["resNumber"]
            currResidueMols = 1
    topologyMoleculesDataFrame = pd.concat(
        [
            topologyMoleculesDataFrame,
            pd.DataFrame(
                {
                    "resName": currResidueName,
                    "mols": currResidueMols,
                },
                index=[0],
            ),
        ],
        ignore_index=True,
    )
    stdout.write("[ molecules ]\n")
    stdout.write("; Compound\t#Mols\n")
    for _idx, row in topologyMoleculesDataFrame.iterrows():
        stdout.write(
            "{resName}\t\t{nMols}\n".format(
                resName=row["resName"], nMols=row["mols"]
            )
        )
    return 0


def run() -> None:
    import argparse
    from sys import exit

    parser = argparse.ArgumentParser(
        description="Genera la entrada '[molecules]' para un archivo de"
        " topología de Gromacs"
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
