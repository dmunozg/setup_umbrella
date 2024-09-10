import os
import shutil
import tempfile
import unittest
from pathlib import Path

from loguru import logger

from setup_umbrella.setupUmbrella import (
    check_file,
    check_group,
    clean_up,
    count_frames,
    generate_pull_tpr,
    separate_trajectory,
)


class TestCheckFileFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.logger = logger

    def test_file_exist_and_hast_correct_extension(self) -> None:
        file_path = os.path.join(self.temp_dir, "test.txt")
        with open(file_path, "w") as _f:
            pass
        self.assertFalse(check_file(file_path, ".txt"))

    def test_file_exist_but_has_incorrect_extension(self) -> None:
        file_path = os.path.join(self.temp_dir, "not_csv.txt")
        with open(file_path, "w") as _f:
            pass
        self.assertTrue(check_file(file_path, ".csv"))

    def test_file_does_not_exist(self) -> None:
        file_path = os.path.join(self.temp_dir, "nonexistent.txt")
        self.assertTrue(check_file(file_path, ".txt"))

    def tearDown(self) -> None:
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)


class TestSeparateTrajectory(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path("tests/test_data")
        self.tpr_file = self.test_dir / "sample.tpr"
        self.traj_file = self.test_dir / "sample.xtc"
        self.work_dir = self.test_dir / "output"
        self.logger = logger

    def test_separate_trajectory(self) -> None:
        result = separate_trajectory(
            tpr_file=str(self.tpr_file),
            trr_file=str(self.traj_file),
            work_dir=str(self.work_dir),
        )
        # Check if trjconv was succesful
        self.assertEqual(result, 0)

        # Check if the expected output files were created
        frame_files = [
            f
            for f in os.listdir(self.work_dir / "coords")
            if f.startswith("conf")
        ]
        self.assertGreater(len(frame_files), 200)

    def test_separate_trajectory_with_invalid_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            separate_trajectory(
                tpr_file="nonexistent.tpr",
                trr_file="nonexistent.xtc",
                work_dir=str(self.work_dir),
            )

    def tearDown(self) -> None:
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)


class TestGeneratePullTpr(unittest.TestCase):
    def setUp(self) -> None:
        test_data_path = Path("tests/test_data")
        self.mdp_file = test_data_path / "working.mdp"
        self.bad_mdp_file = test_data_path / "bad.mdp"
        self.topol_file = test_data_path / "topol.top"
        self.coords_dir = test_data_path / "test_coords"
        self.index_file = test_data_path / "index.ndx"
        self.conf_index = 174
        return super().setUp()

    def test_generate_pull_tpr(self) -> None:
        resulting_tpr_file, resulting_stderr = generate_pull_tpr(
            mdp_file=self.mdp_file,
            topology_file=self.topol_file,
            coords_dir=self.coords_dir,
            configuration_index=self.conf_index,
            index_file=self.index_file,
        )

        self.assertTrue(resulting_tpr_file.exists())
        self.assertIsInstance(resulting_stderr, str)

    def test_generate_pull_tpr_with_bad_mdp(self) -> None:
        with self.assertRaises(RuntimeError):
            generate_pull_tpr(
                mdp_file=self.bad_mdp_file,
                topology_file=self.topol_file,
                coords_dir=self.coords_dir,
                configuration_index=self.conf_index,
                index_file=self.index_file,
            )

    def test_generate_pull_tpr_with_missing_inputs(self) -> None:
        with self.assertRaises(FileNotFoundError):
            generate_pull_tpr(
                mdp_file="nonexistent.mdp",
                topology_file=self.topol_file,
                coords_dir=self.coords_dir,
                configuration_index=self.conf_index,
                index_file=self.index_file,
            )


class TestCountFrames(unittest.TestCase):
    def test_count_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(10):
                filename = f"conf{i}.gro"
                open(Path(temp_dir) / filename, "w").close()
            self.assertEqual(count_frames(temp_dir), 10)

    def test_count_frames_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(count_frames(temp_dir), 0)

    def test_count_frames_no_gro_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(10):
                filename = f"conf{i}.txt"
                open(Path(temp_dir) / filename, "w").close()
            self.assertEqual(count_frames(temp_dir), 0)

    def test_count_frames_invalid_path(self) -> None:
        with self.assertRaises(OSError):
            count_frames("nonexistent/path")


class TestCheckGroup(unittest.TestCase):
    def setUp(self) -> None:
        self.ndx_file = tempfile.NamedTemporaryFile(mode="w")  # noqa: SIM115
        self.group_name = "test_group"

    def tearDown(self) -> None:
        self.ndx_file.close()
        if Path(self.ndx_file.name).exists():
            Path(self.ndx_file.name).unlink()
        return super().tearDown()

    def test_group_found(self) -> None:
        self.ndx_file.write(f"[ {self.group_name} ]\n1\n")
        self.ndx_file.flush()

        result = check_group(self.ndx_file.name, self.group_name)
        self.assertTrue(result)

    def test_group_not_found(self) -> None:
        self.ndx_file.write(f"[ {self.group_name} ]\n1\n")
        self.ndx_file.flush()

        result = check_group(self.ndx_file.name, "nonexistent_group")
        self.assertFalse(result)

    def test_invalid_ndx_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            check_group("nonexistent.ndx", self.group_name)


class TestCleanUp(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.temp_dir.mkdir(exist_ok=True)
        for i in range(10):
            filename = f"dist{i}.xvg"
            open(self.temp_dir / filename, "w").close()

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        return super().tearDown()

    def test_clean_up_success(self) -> None:
        result = clean_up(self.temp_dir)
        self.assertEqual(result, 0)

    def test_clean_up_nonexistent_dir(self) -> None:
        result = clean_up("nonexistent/path")
        self.assertNotEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
