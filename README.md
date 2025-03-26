# DVS Camera Simulator

A simple Dynamic Vision Sensor (DVS) camera simulator that processes videos to create event-based outputs that mimic how DVS cameras capture motion.

## Overview

This project simulates the behavior of a DVS camera by:

1. Reading a video input
2. Converting frames to grayscale
3. Computing differences between consecutive frames
4. Visualizing the differences where:
   - Negative changes become brighter
   - Positive changes become darker
   - No change remains neutral gray (128)

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- UV (for running the script)

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/sbalk/dvs-camera-simulator.git
   cd dvs-camera-simulator
   ```

2. Install UV (Universal Virtualenv):
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Usage

### Preparing Input Videos

1. Place your video files in the `input/` directory (the directory will be created if it doesn't exist)

### Running the Simulator

Run the main script with UV:

```
uv run main.py
```

This will set up an environment when it's ran for the first time and then process all videos in the `input/` directory and save the DVS-simulated outputs to the `output/` directory. It will skip videos that were previously converted.

### Command Line Options

```
uv run main.py --help
```

Available options:

- `--input_dir`: Directory containing input video files (default: input)
- `--output_dir`: Directory to save output video files (default: output)
- `--input_video`: Specific input video file to process
- `--output_video`: Output filename for the processed video
- `--grey_value`: Base gray value for the DVS output (default: 128)
- `--no_display`: Disable display of frames during processing

### Examples

Process a specific video:

```
uv run main.py --input_video input/my_video.mp4 --output_video output/dvs_result.mp4
```

Process all videos with custom settings:

```
uv run main.py --grey_value 150 --no_display
```

## How It Works

The DVS simulator detects changes in pixel intensity between consecutive frames:

- When a pixel gets darker, it appears darker in the output
- When a pixel gets brighter, it appears brighter in the output
- Pixels with no change remain at the neutral gray value

This approach mimics how real DVS cameras only register changes in luminance rather than absolute brightness values.

## File Structure

- `main.py`: Main entry point with command-line interface
- `dvs_simulator.py`: Core DVS simulation implementation
- `input/`: Directory for input videos
- `output/`: Directory for processed DVS videos
