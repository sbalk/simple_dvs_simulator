# Simple DVS Emulator

A simple Dynamic Vision Sensor (DVS) emulator that processes normal video data and generates event-based output similar to a DVS camera.

## Overview

This project simulates the behavior of a DVS camera by:

1. Taking standard video input
2. Converting frames to grayscale
3. Computing pixel differences between consecutive frames
4. Generating event frames where significant changes occurred

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- UV (for dependency management)

## Installation

1. Clone this repository
2. Install dependencies using UV:
   ```
   uv pip install -e .
   ```

## Usage

### Preparing Input Data

1. Place your video files in the `input/` directory

### Running the Simulator

Run the main script to process all videos in the input directory:

```
python main.py
```

This will generate DVS output videos in the `output/` directory.

### Viewing Results

Use the viewer to see original and DVS output side by side:

```
python viewer.py
```

Or specify a specific video:

```
python viewer.py --video your_video.mp4
```

## How It Works

The DVS simulator detects changes in pixel intensity between consecutive frames. When the absolute difference exceeds a threshold (default: 10), it registers an event at that pixel location.

The resulting output shows only the pixels that have changed significantly, similar to how a real DVS camera would detect changes in the scene.

## Customization

You can modify the threshold value in `dvs_simulator.py` to adjust the sensitivity of the DVS emulation.

## File Structure

- `main.py`: Main entry point
- `dvs_simulator.py`: Core DVS simulation logic
- `viewer.py`: Tool for viewing original and DVS videos side by side
- `input/`: Directory for input videos
- `output/`: Directory for output DVS videos
