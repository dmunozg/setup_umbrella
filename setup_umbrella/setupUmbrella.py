#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import re
from pathlib import Path
from sys import exit, stderr
from time import time
from typing import Optional

import gromacs as gmx  # type:ignore
import numpy as np
from loguru import logger


def check_file(file: str, extension: str) -> bool:
    """
    Check if a file exists and has the correct extension.

    Args:
        file (str): The path to the file to be checked. extension (str): The
        expected file extension.

    Returns:
        bool: True if the file does not exist or has an incorrect extension,
        False otherwise.

    Notes:
        This function checks for the existence of a file and its extension. If
        either check fails, it logs an error message and returns True.
        Otherwise, it logs a debug message and returns False.
    """
    # Check if the file exists
    if not os.path.isfile(file):
        logger.critical(f"Could not find {file}")
        return True

    # Check if the file has the correct extension
    if not file.endswith(extension):
        logger.critical(f"{file} is not a {extension} file.")
        return True

    # If both checks pass, log a debug message and return False
    logger.debug(f"{file} will be used as {extension}")
    return False


def separate_trajectory(
    tpr_file: str | Path, trr_file: str | Path, work_dir: str
) -> int:
    """
    Disassemble the trajectory into multiple frames.

    This function uses GROMACS's `trjconv` tool to split a single trajectory
    file into individual frame files. Each frame is written as a separate .gro
    structure file, named "conf.<frame_index>.gro".

    Args:
        tpr_file (str): The path to the topology file (.tpr). trr_file (str):
        The path to the trajectory file (.trr or .xtc). work_dir (str): The
        working directory where frame files will be written.

    Returns:
        int: 0 if successful, non-zero otherwise.
    """
    tpr_file = Path(tpr_file)
    trr_file = Path(trr_file)
    # Check if both input files exist
    if not all([tpr_file.is_file(), trr_file.is_file()]):
        logger.critical("Could not find one or more input files")
        raise FileNotFoundError
    # Create a Path object for the working directory
    workdir_path = Path(work_dir)
    frames_path = workdir_path / "coords"
    frames_path.mkdir(parents=True, exist_ok=True)
    # Use GROMACS's trjconv to split the trajectory into individual frames
    trjconv_runner = gmx.trjconv
    trjconv_runner.faluremode = None
    trjconv_result = trjconv_runner.run(
        s=tpr_file,
        f=trr_file,
        o=str(frames_path / "conf.gro"),
        sep=True,
        input=("System"),
        stdout=False,
        stderr=False,
    )

    # Check if trjconv was successful
    if trjconv_result[0] == 0:
        return 0
    else:
        logger.error("Error in execution of GROMACS trjconv")
        logger.error(trjconv_result[2])
        return 1


def generate_pull_tpr(
    mdp_file: str | Path,
    topology_file: str | Path,
    coords_dir: str | Path,
    configuration_index: int,
    index_file: Optional[str | Path] = None,
) -> tuple[Path, str]:
    """
    Generate a new .tpr file for GROMACS using the provided MDP and topology
    files.

    This function uses the `grompp` tool from GROMACS to generate a new .tpr
    file. It takes in several inputs: an MDP file, a topology file, a directory
    containing coordinate files, and an index specifying which configuration to
    use. An optional index file can also be provided.

    Args:
        mdp_file (str | Path): The path to the MDP file. topology_file (str |
        Path):
            The path to the topology file. coords_dir (str | Path): The
            directory containing coordinate files. configuration_index (int):
            The index of the configuration to use. index_file (Optional[str |
            Path]): An optional index file.

    Returns:
        tuple[Path, str]: A tuple containing the path to the new .tpr file and
        a string representing any output from the `grompp` tool.
    """
    # Ensure every input is a Path object
    mdp_file = Path(mdp_file)
    topology_file = Path(topology_file)
    coords_dir = Path(coords_dir)
    if index_file is not None:
        index_file = Path(index_file)

    # Check if the input files exist
    if not all([
        mdp_file.is_file(),
        topology_file.is_file(),
        coords_dir.is_dir(),
    ]):
        logger.critical("Could not find one or more input files")
        raise FileNotFoundError

    # If an index file is provided, check its existence as well
    if index_file is not None and not index_file.is_file():
        logger.critical("Could not find index file")
        raise FileNotFoundError

    grompp_runner = gmx.grompp
    grompp_runner.failuremode = None

    resulting_tpr_file = coords_dir / f"dist{configuration_index}.tpr"

    # Define keyword arguments for GROMACS grompp
    grompp_kwargs = {
        "f": str(mdp_file),
        "p": str(topology_file),
        "o": str(resulting_tpr_file),
        "r": str(coords_dir / f"conf{configuration_index}.gro"),
        "c": str(coords_dir / f"conf{configuration_index}.gro"),
        "po": str(coords_dir / f"mdout{configuration_index}.mdp"),
        "maxwarn": 1,
    }

    # If an index file is provided, add it to the keyword arguments
    if index_file is not None:
        grompp_kwargs["n"] = str(index_file)

    # Execute GROMACS grompp and check its result
    grompp_result = grompp_runner.run(
        **grompp_kwargs,
        stdout=False,
        stderr=False,
    )
    if grompp_result[0] == 0:
        return (resulting_tpr_file, str(grompp_result[2]))
    else:
        logger.error("Error in execution of GROMACS grompp")
        logger.error(grompp_result[2])
        raise RuntimeError


def count_frames(path: str | Path) -> int:
    """
    Counts the number of frames extracted from trjconv.

    This function takes a path as input, and returns the number of files with
    the pattern "conf*.gro" in that directory. These files are assumed to be
    the output of GROMACS' trjconv command.

    Args:
        path (str): The directory containing the extracted frames.

    Returns:
        int: The number of frames found in the specified directory.
    """
    gro_files_in_path = [f for f in os.listdir(path) if f.endswith(".gro")]
    conf_files_in_path = [f for f in gro_files_in_path if f.startswith("conf")]
    return len(conf_files_in_path)


def check_group(ndx_file: str, group: str) -> bool:
    """
    Checks if a specified group exists in a given GROMACS index file.
    Args:
        ndx_file (str): The path to the GROMACS index file.
        group (str): The name of the group to be checked.

    Returns:
        bool: True if the group is found, False otherwise.

    Raises:
        FileNotFoundError: If the specified ndx_file does not exist.
    """
    try:
        # Attempt to open the index file
        with open(ndx_file) as indexFile:
            # Iterate over each line in the file
            for line in indexFile:
                # Check if the group is present in the current line
                if f"[ {group} ]" in line:
                    # If found, return True
                    return True
        # If the loop completes without finding the group, log an error and
        # return False
        logger.warning(f"Could not find group {group} in index file.")
        return False
    except FileNotFoundError:
        # Log an error if the ndx_file is not found and re-raise the exception
        logger.error(f"The ndx_file '{ndx_file}' was not found")
        raise


def clean_up(directory: str | Path) -> int:
    """
    Deletes all files named "dist<integer>.xvg" in the specified directory.

    Args:
        directory (str): The path to the directory where files will be deleted.

    Returns:
        int: 0 on success, non-zero on failure.
    """
    try:
        for filename in os.listdir(directory):
            if re.match(r"dist\d+\.xvg", filename):
                filepath = os.path.join(directory, filename)
                os.remove(filepath)
                logger.info(f"Removed file {filename}")
    except FileNotFoundError as err:
        logger.error(f"Failed to clean up files: {err}")
        return 1
    except Exception as err:
        logger.error(f"An error occurred during cleanup: {err}")
        return 1
    return 0


def read_distance_worker(
    ID: int,
    inQueue: "mp.Queue[int]",
    outQueue: "mp.Queue[tuple[int, float]]",
    gromppRequirements: dict[str, str],
    workDir: str,
) -> None:
    """Worker that reads the distance out of a dist file and returns a
    tuple with the frame and the measured distance.
    """
    while True:
        if inQueue.empty():
            logger.debug(
                "Queue is empty, Shutting down" f" worker {os.getpid()}"
            )
            return None
        currentFrame: int = inQueue.get()
        logger.debug(
            f"Worker {os.getpid()} got frame {currentFrame}"
            " to read a distance."
        )
        _resulting_tpr, grompp_stderr = generate_pull_tpr(
            mdp_file=gromppRequirements["mdpFile"],
            topology_file=gromppRequirements["topolFile"],
            coords_dir=Path(workDir) / "coords",
            configuration_index=currentFrame,
            index_file=gromppRequirements["indexFile"],
        )
        distancePattern = re.compile(
            r"[\ ]*[\d]{1,2}[\s]*[\d]{1,5}[\s]*[\d]{1,5}[\s]+"
            r"(?P<distance>[\d.-]+)\snm[\s]*[-.\d]{5,6}\snm"
        )

        if distancePattern.findall(grompp_stderr):
            measuredDistance: float = float(
                distancePattern.findall(grompp_stderr)[0]
            )
        else:
            measuredDistance = np.nan
            logger.warning(
                f"Distance could not be measured in frame {currentFrame}"
            )
            dumpFileDestination: str = os.path.join(
                workDir, "coords", f"dist{currentFrame}.out"
            )
            with open(dumpFileDestination, "w") as dumpFile:
                print(grompp_stderr, file=dumpFile, end="")
        outQueue.put((currentFrame, measuredDistance))


def list_and_sort(
    queue: "mp.Queue[tuple[int, float]]",
) -> list[tuple[int, float]]:
    """Converts a queue of tuples to a sorted list."""
    unsorted_list: list[tuple[int, float]] = []
    while not queue.empty():
        item: tuple[int, float] = queue.get()
        unsorted_list.append(item)
    # Use the built-in sort function with a lambda key for efficient sorting
    unsorted_list.sort(key=lambda tup: tup[0])
    return unsorted_list


def find_windows(
    frameList: list[tuple[int, float]], interval: float
) -> list[tuple[int, float]]:
    """Returns a list of the starting configurations that should be used
    in the umbrella sampling"""
    selectedFrames: list[tuple[int, float]] = []
    currentIndex: int = 0
    finished: bool = False
    while not finished:
        currentFrame: tuple[int, float] = frameList[currentIndex]
        selectedFrames.append(currentFrame)
        for frame in frameList[currentIndex + 1 :]:
            distanceFromInterval: float = (
                abs(frame[1] - currentFrame[1]) - interval
            )
            if frame[0] == frameList[-1][0]:
                finished = True
                logger.debug("Reached final frame. Finishing.")
                break
            if distanceFromInterval >= 0:
                logger.debug(
                    f"Frame {frame[0]} has a {distanceFromInterval} distance"
                    f" from frame {currentFrame[0]}"
                )
                currentFrame = frame
                currentIndex = frameList.index(frame)
                break
    return selectedFrames


def check_interval(interval: str) -> int:
    """Checks if the given interval is a valid number."""
    try:
        float_interval: float = float(interval)
    except ValueError:
        logger.error(f"Typed interval ({interval}) is not a number")
        return 1
    if float_interval == 0.0:
        logger.error(f"Window interval ({interval}) cannot be 0")
        return 1
    return 0


def filter_out_by_limits(
    distancesList: list[tuple[int, float]], minLim: float, maxLim: float
) -> list[tuple[int, float]]:
    logger.debug(
        f"Will discard every frame that is not between {minLim} and {maxLim}"
    )
    newDistancesList: list[tuple[int, float]] = []
    for distanceID, distanceValue in distancesList:
        if (distanceValue >= minLim) and (distanceValue <= maxLim):
            newDistancesList.append((distanceID, distanceValue))
        else:
            pass
    logger.info(
        f"{len(distancesList) - len(newDistancesList)} frames were discarded."
    )
    return newDistancesList


def main(
    mdp_file: str,
    trajectory_file: str,
    index_file: str,
    tpr_file: str,
    topol_file: str,
    group_a: str,
    group_b: str,
    output_dir: str,
    windows_spacing: float,
    limits: Optional[tuple[float, float]] = None,
) -> int:
    # Check if all specified files exist
    if check_file(mdp_file, ".mdp"):
        return 1
    if check_file(trajectory_file, ".xtc") and check_file(
        trajectory_file, ".trr"
    ):
        return 1
    if check_file(index_file, ".ndx"):
        return 1
    if check_file(tpr_file, ".tpr"):
        return 1
    if check_file(topol_file, ".top"):
        return 1
    # Check if the specified groups are inside the index file.
    if not check_group(index_file, group_a):
        return 1
    if not check_group(index_file, group_b):
        return 1
    # Check if separation has already been done
    confirmOverwrite = False
    if os.path.isdir(os.path.join(output_dir, "coords")):
        logger.info("Trajectory splitting appears to be already done.")
        confirmOverwriteInput = input("Do you want to overwrite it?[y/N]]")
        confirmOverwrite = confirmOverwriteInput != "y"
    # Create the working directory if they dont exist
    Path(os.path.join(output_dir, "coords")).mkdir(parents=True, exist_ok=True)
    # Divide trajectory in multiple frames if necessary
    if not confirmOverwrite:
        logger.debug(
            "Separating " + trajectory_file + " into multiple configurations."
        )
        timeCheck = time()
        if (
            separate_trajectory(tpr_file, trajectory_file, work_dir=output_dir)
            == 0
        ):
            pass
        else:
            logger.critical(
                "There has been a problem when separating the" "trajectory."
            )
            return 1
        elapsed = round(time() - timeCheck, 3)
        logger.info(f"Took {elapsed} seconds to separate the trajectory")
    # Count the number of frames in coords folder
    totalConf = count_frames(os.path.join(output_dir, "coords"))
    logger.info(f"{totalConf} frames found in coords directory.")
    # Dont redo this if we're not overwriting
    if os.path.isfile(os.path.join(output_dir, "summary_distances.dat")):
        isFirstRun = False
        logger.info("Distances measurement seems to already been done.")
        confirmOverwriteInput = input("Do you want to overwrite it?[y/N]]")
        if confirmOverwriteInput.upper() == "Y":
            confirmOverwrite = True
        else:
            confirmOverwrite = False
    else:
        isFirstRun = True
        confirmOverwrite = False
    if confirmOverwrite or isFirstRun:
        # Calculate the distance in each frame.
        # We need the necessary files to launch a grompp run:
        timeCheck = time()
        gromppRequirements = {
            "mdpFile": mdp_file,
            "indexFile": index_file,
            "topolFile": topol_file,
        }
        logger.info("Measuring distances from calculation... ")
        # Refill the queue
        framesQueue: mp.Queue[int] = mp.Queue()
        resultsQueue: mp.Queue[tuple[int, float]] = mp.Queue()
        for frame in range(totalConf):
            framesQueue.put(frame)
            workerProcesses = []
        for workerID in range(mp.cpu_count()):
            workerProcess = mp.Process(
                target=read_distance_worker,
                args=(
                    workerID,
                    framesQueue,
                    resultsQueue,
                    gromppRequirements,
                    output_dir,
                ),
            )
            workerProcess.start()
            workerProcesses.append(workerProcess)
        # TODO: Sustituir esto por una barra en tqdm
        # while not framesQueue.empty():
        #     sleep(0.1)
        #     print(
        #         f"Remaining frames: {framesQueue.qsize()} \t"
        #         f" Completed frames: {resultsQueue.qsize()} ",
        #         end="\r",
        #         file=stdout,
        #     )

        for workerProcess in workerProcesses:
            logger.debug(f"Waiting for worker {workerProcess.pid} to finish.")
            workerProcess.join(2)
            workerProcess.terminate()
        elapsed = round(time() - timeCheck, 3)
        logger.info(f"Took {elapsed} seconds to calculate every distance.")

        # Convert resulting queue to a list.
        distancesList = list_and_sort(resultsQueue)
        logger.debug(
            f"Resulting distance list has {len(distancesList)} distances"
        )
        # Delete previous list if we're overwriting
        if confirmOverwrite:
            os.remove(os.path.join(output_dir, "summary_distances.dat"))
        # Print resulting distances to a file.
        logger.debug(
            "Writing distances to",
            os.path.join(output_dir, "summary_distances.dat"),
        )
        with open(
            os.path.join(output_dir, "summary_distances.dat"), "w"
        ) as summaryFile:
            for distance in distancesList:
                print(distance[0], distance[1], sep="\t", file=summaryFile)

    # If this is not the first run, or we're not overwriting we should read
    # from the already generated file
    else:
        with open(
            os.path.join(output_dir, "summary_distances.dat")
        ) as summaryFile:
            distancesList = []
            for line in summaryFile:
                lineContent = line.split()
                distancesList.append((
                    int(lineContent[0]),
                    float(lineContent[1]),
                ))
        logger.info(
            f"{len(distancesList)} frames found in summary_distances.dat file"
        )
    # Set interval if it was specified at runtime
    if windows_spacing > 0.0:
        windows_spacing = abs(windows_spacing)
        noConfirm = True
    else:
        noConfirm = False
    while True:
        if noConfirm is False:
            interval = input("Enter the desired window interval: ")
            if check_interval(interval):
                continue
            windows_spacing = abs(float(interval))
        # Check if limits were set
        if limits:
            distancesList = filter_out_by_limits(
                distancesList, limits[0], limits[1]
            )
            # Sort the resulting distances list by distance
            distancesList.sort(key=lambda tup: tup[1])
            logger.info(
                f"{len(distancesList)} frames remained after a limit filter."
            )
            confirmShowFilteredDistances = input(
                "Do you want to see the filtered distances list? [y/N]: "
            )
            if confirmShowFilteredDistances.upper() == "Y":
                for frameData in distancesList:
                    logger.info("{}\t\t{}".format(*frameData))
                else:
                    pass
        logger.info("Building window list...")
        windowList = find_windows(distancesList, windows_spacing)
        logger.info("Frame \tDistance \tFrom Interval")
        lastFrame = windowList[0]
        logger.info(f"{lastFrame[0]}\t{round(lastFrame[1], 4)}\t\t{0.0}")
        for selectedFrame in windowList[1:]:
            logger.info(
                f"{selectedFrame[0]}\t{round(selectedFrame[1], 4)}\t\t"
                f"{round(selectedFrame[1] - lastFrame[1], 4)}"
            )
            lastFrame = selectedFrame
        if noConfirm is False:
            confirmSelection = input(
                "Do you wish to keep this results?[Y/n]: "
            )
            if confirmSelection == "n":
                continue
            else:
                break
        break
    # Write selected frames to file.
    logger.debug(
        "Writing selected frames to",
        os.path.join(output_dir, "selected_frames.dat"),
    )
    with open(
        os.path.join(output_dir, "selected_frames.dat"), "w"
    ) as selectedFramesFile:
        for frame, _distance in windowList:
            print(frame, file=selectedFramesFile)
    logger.debug("Finished")
    return 0


@logger.catch
def run() -> None:
    # Define the program arguments
    desc = "Sets up an umbrella calculation from a pull simulation."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-f",
        "--mdp",
        help="MDP file to be used in the umbrella calculations",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--trajectory",
        help="Pull trajectory file. (.xtc or .trr)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--tpr",
        help="Portable binary input file (.tpr)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n", "--index", help="Index file (.ndx)", type=str, required=True
    )
    parser.add_argument(
        "-p",
        "--topol",
        metavar="topol.top",
        default="topol.top",
        help="Topology file (.top)",
        type=str,
        required=True,
    )
    # TODO: These groups can be deduced from the mdp file. They should not be
    # required
    parser.add_argument(
        "-A", "--groupA", help="Name of group A", type=str, required=True
    )
    parser.add_argument(
        "-B", "--groupB", help="Name of group B", required=True
    )
    parser.add_argument(
        "--limit",
        nargs=2,
        type=float,
        help="Ignore windows outside these limits",
        default=None,
    )
    parser.add_argument(
        "-sp",
        help="spacing between windows. Will ask by default",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-u",
        metavar="umbrella/",
        help="Folder where the resulting files will be left",
        type=str,
        default="umbrella",
    )
    parser.add_argument("-v", help="Be verbose", action="store_true")
    # Parse the arguments
    args = parser.parse_args()
    # Log all messages to a file
    # Define a handler to print to screen more important messages
    logger.remove()
    if args.v:
        logger.add(stderr, level="DEBUG")
    else:
        logger.add(stderr, level="INFO")
    if args.tpr:
        logger.add(args.tpr.split(".")[0] + ".log", level="DEBUG")
    # Run main function
    exit(
        main(
            mdp_file=args.mdp,
            trajectory_file=args.trajectory,
            index_file=args.index,
            tpr_file=args.tpr,
            topol_file=args.topol,
            group_a=args.groupA,
            group_b=args.groupB,
            output_dir=args.u,
            windows_spacing=args.sp,
            limits=args.limit,
        )
    )


if __name__ == "__main__":
    run()
