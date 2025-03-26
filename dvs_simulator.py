#!/usr/bin/env python3
"""
DVS Camera Simulator

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

import cv2
import numpy as np
import os
from pathlib import Path

# Constants
GREY_VALUE = 128  # Middle grey (8-bit)
INPUT_DIR = Path("output")
OUUTPUTS_DIR = Path("input")


class DVSSimulator:
    """
    A class to simulate a Dynamic Vision Sensor (DVS) camera.
    """

    def __init__(self, input_path, output_path, grey_value=128):
        """
        Initialize the DVS simulator.

        Args:
            input_path: Path to the input video file
            output_path: Path to save the output video
            grey_value: Base gray value for the output (default 128)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.grey_value = grey_value

        # Ensure output directory exists
        os.makedirs(self.output_path.parent, exist_ok=True)

    def process(self):
        """Process the video to create a DVS simulation."""
        # Open the video file
        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file {self.input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(self.output_path), fourcc, fps, (width, height), isColor=False
        )

        # Read the first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read first frame from video")

        # Convert first frame to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_count = 0

        print(f"Processing video: {self.input_path}")
        print(f"Output will be saved to: {self.output_path}")
        print(f"Total frames to process: {total_frames}")

        # Process frame by frame
        while True:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate the difference between frames
            diff = cv2.subtract(curr_gray, prev_gray)

            # Create the DVS visualization
            # Start with a uniform grey frame
            dvs_frame = np.ones_like(curr_gray) * self.grey_value

            # Apply the changes:
            # Subtract diff from grey so that:
            # - Negative changes (darkening) become brighter (> grey_value)
            # - Positive changes (brightening) become darker (< grey_value)
            dvs_frame = cv2.subtract(dvs_frame, diff)

            # Write the frame to output video
            out.write(dvs_frame)

            # Update previous frame
            prev_gray = curr_gray

            # Progress indication
            frame_count += 1
            if frame_count % 50 == 0 or frame_count == total_frames:
                print(
                    f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)"
                )

            # Display the frame (optional)
            cv2.imshow("DVS Simulation", dvs_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Processing complete. Processed {frame_count} frames.")
        print(f"Output saved to: {self.output_path}")

        return self.output_path


def main():
    """Main function to run the DVS camera simulator."""
    simulator = DVSSimulator(
        input_path=INPUT_VIDEO, output_path=OUTPUT_VIDEO, grey_value=GREY_VALUE
    )
    simulator.process()


if __name__ == "__main__":
    main()
