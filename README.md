# Fitness Counter

Real-time exercise repetition counter using computer vision and pose estimation.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange.svg)

## About

This application uses your webcam to detect body poses and automatically count exercise repetitions. It tracks joint angles to determine when a rep is completed.

### Supported Exercises

| Exercise | Tracked Angle |
|----------|---------------|
| Squat | Hip → Knee → Ankle |
| Push-up | Shoulder → Elbow → Wrist |
| Bicep Curl | Shoulder → Elbow → Wrist |

## Demo

The app displays:
- Live skeleton overlay on your body
- Current joint angle
- Rep counter
- Exercise mode indicator

## Installation

```bash
git clone https://github.com/yourusername/FitnessCounter.git
cd FitnessCounter

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
python3 fitness_counter.py
```

### Controls

| Key | Action |
|-----|--------|
| `1` | Squat mode |
| `2` | Push-up mode |
| `3` | Bicep curl mode |
| `R` | Reset counter |
| `Q` | Quit |

## How It Works

1. **Pose Detection** - MediaPipe BlazePose extracts 33 body landmarks from the webcam feed
2. **Angle Calculation** - Computes the angle between three key joints using arctangent
3. **Rep Counting** - Detects up/down transitions based on angle thresholds

```
Standing (angle > 160°) → Squatting (angle < 90°) → count + 1
```

## Requirements

- Python 3.9+
- Webcam
- macOS / Linux / Windows

## License

MIT
