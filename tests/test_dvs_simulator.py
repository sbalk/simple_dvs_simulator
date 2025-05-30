"""Test module for the DVS simulator functionality."""

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Add the parent directory to the Python path to allow importing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modules to test
import main
from dvs_simulator import DVSSimulator

# Constants for tests to avoid magic values
TEST_GREY_VALUE = 128
TEST_FRAME_WIDTH = 64
TEST_FRAME_HEIGHT = 48
TEST_FRAME_COUNT = 150


class TestDVSSimulator:
    """Test class for the DVS simulator functionality."""

    @pytest.fixture()
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for testing.

        Returns
        -------
            Generator yielding a temporary directory path

        """
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture()
    def mock_video(self, temp_dir: Path) -> Path:
        """Create a small test video file.

        Args:
        ----
            temp_dir: Temporary directory to create the video in

        Returns:
        -------
            Path to the created test video

        """
        # Create a small video file for testing
        video_path = temp_dir / "test_video.mp4"
        # Create a simple 10-frame grayscale video
        width, height = TEST_FRAME_WIDTH, TEST_FRAME_HEIGHT
        fps = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(video_path),
            fourcc,
            fps,
            (width, height),
            isColor=False,
        )

        for i in range(10):
            # Create a gradient frame
            frame = np.ones((height, width), dtype=np.uint8) * (i * 25)
            out.write(frame)

        out.release()

        return video_path

    def test_ensure_directory(self, temp_dir: Path) -> None:
        """Test that ensure_directory creates directories when they don't exist.

        Args:
        ----
            temp_dir: Temporary directory for testing

        """
        test_dir = temp_dir / "test_dir"
        assert not test_dir.exists()

        main.ensure_directory(test_dir)
        assert test_dir.exists()
        assert test_dir.is_dir()

        # Test that it doesn't raise errors when directory already exists
        main.ensure_directory(test_dir)
        assert test_dir.exists()

    @patch("cv2.VideoCapture")
    @patch("cv2.VideoWriter")
    def test_process_video(
        self,
        mock_writer: MagicMock,
        mock_capture: MagicMock,
        temp_dir: Path,
    ) -> None:
        """Test the process_video function with mocks.

        Args:
        ----
            mock_writer: Mock for VideoWriter
            mock_capture: Mock for VideoCapture
            temp_dir: Temporary directory for testing

        """
        input_path = temp_dir / "test_input.mp4"
        output_path = temp_dir / "test_output.mp4"

        # Configure mocks
        mock_capture_instance = MagicMock()
        mock_capture.return_value = mock_capture_instance
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)

        # Mock reading frames
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * TEST_FRAME_COUNT

        # Return True, frame1 for first call, then True, frame2, then False, None
        mock_capture_instance.read.side_effect = [
            (True, frame1),
            (True, frame2),
            (False, None),
        ]

        # Test the function
        result = main.process_video(input_path, output_path, display=False)

        # Verify the result and calls
        assert result is True
        mock_capture.assert_called_once_with(str(input_path))
        mock_writer.assert_called_once()
        assert mock_writer().write.call_count == 1  # Only one frame difference is written

    def test_dvs_simulator_init(self, temp_dir: Path) -> None:
        """Test DVSSimulator initialization.

        Args:
        ----
            temp_dir: Temporary directory for testing

        """
        input_path = temp_dir / "input.mp4"
        output_path = temp_dir / "output.mp4"

        simulator = DVSSimulator(input_path, output_path, grey_value=TEST_FRAME_COUNT)

        assert simulator.input_path == input_path
        assert simulator.output_path == output_path
        assert simulator.grey_value == TEST_FRAME_COUNT

        # Check that output directory was created
        assert output_path.parent.exists()

    @patch("cv2.VideoCapture")
    @patch("cv2.VideoWriter")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    @patch("cv2.destroyAllWindows")
    def test_dvs_simulator_process(
        self,
        mock_destroy: MagicMock,  # noqa: ARG002
        mock_wait: MagicMock,  # noqa: ARG002
        mock_show: MagicMock,  # noqa: ARG002
        mock_writer: MagicMock,
        mock_capture: MagicMock,
        temp_dir: Path,
    ) -> None:
        """Test the DVSSimulator process method.

        Args:
        ----
            mock_destroy: Mock for destroyAllWindows (unused)
            mock_wait: Mock for waitKey (unused)
            mock_show: Mock for imshow (unused)
            mock_writer: Mock for VideoWriter
            mock_capture: Mock for VideoCapture
            temp_dir: Temporary directory for testing

        """
        input_path = temp_dir / "test_input.mp4"
        output_path = temp_dir / "test_output.mp4"

        # Configure mocks
        mock_capture_instance = MagicMock()
        mock_capture.return_value = mock_capture_instance
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)

        # Mock reading frames
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * TEST_FRAME_COUNT

        # Return True, frame1 for first call, then True, frame2, then False, None
        mock_capture_instance.read.side_effect = [
            (True, frame1),
            (True, frame2),
            (False, None),
        ]

        # Create simulator and test process method
        simulator = DVSSimulator(input_path, output_path)

        # Test should pass without exceptions
        simulator.process()

        # Verify mock calls
        mock_capture.assert_called_once_with(str(input_path))
        mock_writer.assert_called_once()
        assert mock_writer().write.call_count == 1  # Only one frame difference is written

    @patch("argparse.ArgumentParser.parse_args")
    @patch("main.process_video")
    @patch("main.process_all_videos")
    def test_main_function_single_video(
        self,
        mock_process_all: MagicMock,
        mock_process_video: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        """Test the main function with a single video argument.

        Args:
        ----
            mock_process_all: Mock for process_all_videos
            mock_process_video: Mock for process_video
            mock_args: Mock for parse_args

        """
        # Mock arguments for processing a single video
        args = MagicMock()
        args.input_dir = "input"
        args.output_dir = "output"
        args.input_video = "test.mp4"
        args.output_video = None
        args.grey_value = TEST_GREY_VALUE
        args.no_display = True
        mock_args.return_value = args

        # Run the main function
        main.main()

        # Verify that process_video was called with expected arguments
        mock_process_video.assert_called_once()
        # Check the first argument (input_video)
        assert mock_process_video.call_args[0][0] == Path("input") / "test.mp4"
        # Second argument should be the output path with "dvs_" prefix
        assert mock_process_video.call_args[0][1] == Path("output") / "dvs_test.mp4"
        # Last argument should be display=False since no_display is True
        assert mock_process_video.call_args[0][3] is False

        # Verify process_all_videos was not called
        mock_process_all.assert_not_called()

    @patch("argparse.ArgumentParser.parse_args")
    @patch("main.process_video")
    @patch("main.process_all_videos")
    def test_main_function_all_videos(
        self,
        mock_process_all: MagicMock,
        mock_process_video: MagicMock,
        mock_args: MagicMock,
    ) -> None:
        """Test the main function with processing all videos.

        Args:
        ----
            mock_process_all: Mock for process_all_videos
            mock_process_video: Mock for process_video
            mock_args: Mock for parse_args

        """
        # Mock arguments for processing all videos
        args = MagicMock()
        args.input_dir = "input"
        args.output_dir = "output"
        args.input_video = None
        args.output_video = None
        args.grey_value = TEST_GREY_VALUE
        args.no_display = False
        mock_args.return_value = args

        # Run the main function
        main.main()

        # Verify process_all_videos was called with expected arguments
        mock_process_all.assert_called_once()
        assert mock_process_all.call_args[0][0] == Path("input")
        assert mock_process_all.call_args[0][1] == Path("output")
        assert mock_process_all.call_args[0][2] == TEST_GREY_VALUE
        assert mock_process_all.call_args[0][3] is True  # display should be True

        # Verify process_video was not called
        mock_process_video.assert_not_called()

    def test_integration_with_real_video(
        self,
        mock_video: Path,
        temp_dir: Path,
    ) -> None:
        """Integration test with a real (small) video file.

        Args:
        ----
            mock_video: Path to mock video file
            temp_dir: Temporary directory for testing

        """
        output_path = temp_dir / "dvs_output.mp4"

        # Process the video
        result = main.process_video(mock_video, output_path, display=False)

        # Check results
        assert result is True
        assert output_path.exists()

        # Verify the output video has the expected properties
        cap = cv2.VideoCapture(str(output_path))
        try:
            assert cap.isOpened()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert width == TEST_FRAME_WIDTH
            assert height == TEST_FRAME_HEIGHT

            # Should have one fewer frame than input (9 instead of 10)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert frame_count > 0
        finally:
            cap.release()

    def test_process_all_videos(self, temp_dir: Path, mock_video: Path) -> None:
        """Test processing all videos in a directory.

        Args:
        ----
            temp_dir: Temporary directory for testing
            mock_video: Path to mock video file

        """
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"

        # Create input directory with test videos
        input_dir.mkdir()
        output_dir.mkdir()

        # Copy the mock video to the input directory
        shutil.copy(mock_video, input_dir / "video1.mp4")

        # Create another test video
        width, height = TEST_FRAME_WIDTH, TEST_FRAME_HEIGHT
        fps = 30
        video2_path = input_dir / "video2.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(video2_path),
            fourcc,
            fps,
            (width, height),
            isColor=False,
        )

        for i in range(5):
            frame = np.ones((height, width), dtype=np.uint8) * (i * 50)
            out.write(frame)

        out.release()

        # Process all videos
        main.process_all_videos(input_dir, output_dir, display=False)

        # Check that output files were created
        assert (output_dir / "dvs_video1.mp4").exists()
        assert (output_dir / "dvs_video2.mp4").exists()

        # Check that if we run it again, it doesn't reprocess videos
        with patch("main.process_video") as mock_process:
            main.process_all_videos(input_dir, output_dir, display=False)
            # process_video should not be called since videos were already processed
            mock_process.assert_not_called()


if __name__ == "__main__":
    pytest.main()
