#!/usr/bin/env python3
"""DVS Camera Simulator.

This script simulates a Dynamic Vision Sensor (DVS) camera by:
1. Reading a video file
2. Converting frames to grayscale
3. Computing differences between consecutive frames
4. Visualizing the differences where:
   - Negative changes (darkening) become brighter
   - Positive changes (brightening) become darker
   - The background is gray (128)
5. Saving the result to a video file
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

# Constants
GREY_VALUE = 128  # Middle grey (8-bit)
DEFAULT_INPUT_DIR = Path("input")
DEFAULT_OUTPUT_DIR = Path("output")


def ensure_directory(directory: str | Path) -> None:
    """Ensure the directory exists."""
    os.makedirs(directory, exist_ok=True)


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    grey_value: int = GREY_VALUE,
    display: bool = False,
) -> bool:
    """Process the video to simulate DVS camera output.

    Args:
    ----
        input_path: Path to the input video file
        output_path: Path to save the output video
        grey_value: Base gray value for the output (default 128)
        display: Whether to display frames during processing

    Returns:
    -------
        bool: True if processing completed successfully

    """
    # Open the video file
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4v codec
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=False)

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return False

    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0

    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Total frames to process: {total_frames}")

    # Process frame by frame
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the difference
        # Subtract previous from current -> positive values mean the scene got brighter
        diff = cv2.subtract(curr_gray, prev_gray)

        # Create the DVS visualization
        # Start with a grey frame
        dvs_frame = np.ones_like(curr_gray) * grey_value

        # Apply the changes:
        # - Negative changes (curr < prev) become brighter (> 128)
        # - Positive changes (curr > prev) become darker (< 128)
        # We invert diff so that positive changes (brightening) become darker
        dvs_frame = cv2.subtract(dvs_frame, diff)

        # Write the frame
        out.write(dvs_frame)

        # Update previous frame
        prev_gray = curr_gray

        frame_count += 1
        if frame_count % 50 == 0 or frame_count == total_frames:
            print(
                f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)",
            )

        # Display the frame
        if display:
            cv2.imshow("DVS Simulation", dvs_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Clean up
    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()

    print(f"Processing complete. Processed {frame_count} frames.")
    print(f"Output saved to: {output_path}")

    return True


def process_all_videos(
    input_dir: str | Path,
    output_dir: str | Path,
    grey_value: int = GREY_VALUE,
    display: bool = False,
) -> None:
    """Process all video files in the input directory, except those already converted.

    Args:
    ----
        input_dir: Directory containing input videos
        output_dir: Directory to save output videos
        grey_value: Base gray value for the output
        display: Whether to display frames during processing

    """
    # Ensure output directory exists
    ensure_directory(output_dir)

    # Get all video files in the input directory
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
    video_files = []

    for ext in video_extensions:
        video_files.extend(list(Path(input_dir).glob(f"*{ext}")))

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    # Get list of already processed videos (based on filenames)
    processed_videos: set[str] = set()
    for output_file in Path(output_dir).glob("dvs_*.mp4"):
        # Extract the original filename from the output filename (remove 'dvs_' prefix)
        original_name = output_file.stem[4:]  # Skip the 'dvs_' prefix
        processed_videos.add(original_name)

    # Filter out already processed videos
    videos_to_process = [video_file for video_file in video_files if video_file.stem not in processed_videos]

    if not videos_to_process:
        print("All videos have already been processed.")
        return

    print(
        f"Found {len(videos_to_process)} videos to process out of {len(video_files)} total videos.",
    )

    # Process each video
    for video_file in videos_to_process:
        output_file = Path(output_dir) / f"dvs_{video_file.stem}.mp4"
        process_video(video_file, output_file, grey_value, display)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
        argparse.Namespace: The parsed arguments

    """
    parser = argparse.ArgumentParser(description="DVS Camera Simulator")

    # Add arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing input video files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save output video files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default=None,
        help="Specific input video file to process (default: process all videos in input_dir)",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help='Output filename for the processed video (default: "dvs_<input_filename>.mp4")',
    )
    parser.add_argument(
        "--grey_value",
        type=int,
        default=GREY_VALUE,
        help=f"Base gray value for the DVS output (default: {GREY_VALUE})",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable display of frames during processing",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to run the DVS camera simulator."""
    args = parse_arguments()

    # Convert string paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Ensure directories exist
    ensure_directory(input_dir)
    ensure_directory(output_dir)

    # Process single video or all videos
    if args.input_video:
        input_video = Path(args.input_video)
        if not input_video.is_absolute():  # If a relative path is provided
            input_video = input_dir / input_video

        if args.output_video:
            output_video = Path(args.output_video)
            if not output_video.is_absolute():  # If a relative path is provided
                output_video = output_dir / output_video
        else:
            output_video = output_dir / f"dvs_{input_video.stem}.mp4"

        process_video(input_video, output_video, args.grey_value, not args.no_display)
    else:
        process_all_videos(input_dir, output_dir, args.grey_value, not args.no_display)


if __name__ == "__main__":
    main()
