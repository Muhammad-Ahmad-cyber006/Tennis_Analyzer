# 🎾 Tennis Analyzer

An AI-powered tennis match analysis system that detects players and the ball, maps them onto a 2D mini-court, and computes real-time performance stats — shot speed, player speed, and shot count — directly overlaid on the output video.

## Overview

Tennis Analyzer processes a broadcast-style tennis video and produces an annotated version showing:

- 🟢 **Player tracking** — bounding boxes around both players (crowd/umpire filtered out)
- 🟡 **Ball tracking** — bounding box + interpolated trajectory for frames where detection is missed
- 📐 **Court keypoint detection** — court lines detected once from the first frame (fixed camera)
- 🗺️ **Mini court view** — a top-down 2D representation of the court with live player/ball positions
- 📊 **Live stats overlay** — shot speed, average shot speed, and player movement speed for both players

## How It Works

1. **Video ingestion** — the input video is read into a list of frames.
2. **Player detection** — a YOLOv8 model detects and tracks both players frame by frame.
3. **Ball detection** — a custom-trained YOLO model (small/fast objects need a specialized model) detects the ball, with missing frames filled in via interpolation.
4. **Court line detection** — a keypoint model detects the court's line coordinates from the first frame only, since the camera is fixed.
5. **Player filtering** — using the court keypoints, only the two actual players are kept (crowd, ball boys, umpires are discarded).
6. **Mini court mapping** — real pixel coordinates for players and the ball are converted into coordinates on a scaled-down 2D court diagram.
7. **Shot detection** — changes in the ball's direction are used to detect the frames where a shot was struck.
8. **Stats computation** — for each shot-to-shot interval:
   - Ball travel distance (converted from pixels to meters using the real-world doubles line width) and time gives **ball shot speed**.
   - The player closest to the ball at the time of the shot is marked as the shot-taker; the other player's movement in that interval gives **opponent speed**.
   - Stats are merged across all frames and forward-filled, then averaged.
9. **Rendering** — bounding boxes, court keypoints, the mini court, and the stats panel are all drawn onto the output frames, along with a frame counter for debugging.
10. **Output** — the final annotated video is saved to `output_videos/`.

## Project Structure

```
Tennis_Analyzer/
├── analysis/              # Notebooks/scripts for exploratory analysis
├── constants/             # Fixed values (e.g. court dimensions in meters)
├── court_line_detector/   # Court keypoint detection model + inference logic
├── input_videos/          # Source video(s) to analyze
├── mini_court/            # Mini court drawing + coordinate conversion logic
├── models/                # Trained model weights (ball detector, court keypoints)
├── output_videos/         # Generated annotated output videos
├── runs/detect/           # YOLO inference run artifacts
├── tracker_stubs/         # Cached (pickled) detections to speed up repeated runs
├── trackers/              # PlayerTracker and BallTracker classes
├── traning/               # Model training notebooks/scripts
├── utils/                 # Video I/O, distance/conversion helpers, stats drawing
├── main.py                # Main pipeline entry point
├── yolo_inference.py      # Standalone YOLO inference/testing script
└── yolov8x.pt             # Base YOLOv8 model weights (player detection)
```

## Requirements

- Python 3.8+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (`ultralytics`)
- OpenCV (`opencv-python`)
- PyTorch (`torch`)
- pandas
- NumPy

Install the core dependencies:

```bash
pip install ultralytics opencv-python torch torchvision pandas numpy
```

> Model weights (`yolov8x.pt`, `models/yolo5_last.pt`, `models/keypoints_model.pth`) are tracked via **Git LFS** due to their size. Make sure you have [Git LFS](https://git-lfs.com/) installed before cloning:
> ```bash
> git lfs install
> git clone https://github.com/Muhammad-Ahmad-cyber006/Tennis_Analyzer.git
> ```

## Usage

1. Place your input video in `input_videos/` (e.g. `input_videos/input_video.mp4`).
2. Run the pipeline:

```bash
python main.py
```

3. The annotated output video will be saved to `output_videos/output_video.avi`.

### Notes

- On first run, set `read_from_stub=False` in `main.py` for both `player_tracker.detect_frames(...)` and `ball_tracker.detect_frames(...)` so detections are generated fresh instead of loaded from the cached pickle files in `tracker_stubs/`.
- Detection results can be cached to `tracker_stubs/*.pkl` to avoid re-running the (slower) detection models on every test run.

## Models Used

| Component          | Model                                  | Purpose                              |
|---------------------|-----------------------------------------|---------------------------------------|
| Player detection    | YOLOv8x (`yolov8x.pt`)                 | General object detection, filtered to players |
| Ball detection       | Custom YOLOv5 (`models/yolo5_last.pt`) | Fine-tuned for small, fast-moving ball |
| Court keypoints      | Custom keypoint model (`models/keypoints_model.pth`) | Detects court line intersections |
