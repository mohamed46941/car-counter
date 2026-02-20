# Vehicle Tracking and Counting 🚗

This project implements **real-time vehicle detection, tracking, and counting** from video streams using **YOLOv10** for object detection and **DeepSORT** for tracking. It is designed to monitor traffic and count vehicles crossing a virtual line.

---

## Features

* Real-time vehicle detection using YOLOv10.
* Multi-object tracking with **DeepSORT**.
* Automatic counting of vehicles crossing a predefined line.
* Supports multiple vehicle classes (car, bus, truck, etc.).
* Visualizes bounding boxes, track IDs, and count on the video frame.

---

## Requirements

* Python 3.8+
* OpenCV
* ultralytics (YOLOv10)
* deep_sort_realtime
* numpy

Install dependencies:

```bash
pip install opencv-python ultralytics deep_sort_realtime numpy
```

---

## How It Works

1. **Load YOLOv10 model:**
   Detect vehicles in each frame of the input video.

2. **Initialize DeepSORT tracker:**
   Assign unique IDs to detected vehicles and track them across frames.

3. **Detection Filtering:**

   * Only vehicle classes are considered (car, bus, truck, motorcycle, etc.).
   * Detections with confidence lower than 0.3 are ignored.

4. **Counting Vehicles:**

   * A virtual line is drawn across the frame.
   * Vehicles crossing the line from top to bottom are counted.
   * Each vehicle is counted only once using its unique track ID.

5. **Visualization:**

   * Bounding boxes with track ID and class name.
   * Total vehicle count displayed on screen.
   * Real-time video output with detection and tracking overlays.

---

## Usage

1. Set the video path:

```python
cap = cv2.VideoCapture("cars2.mp4")
```

2. Adjust tracking and counting parameters:

* `line_y`: y-coordinate of counting line
* `tracker`: DeepSORT parameters (max_age, n_init, etc.)
* `conf_threshold`: confidence threshold for detection

3. Run the script to visualize tracking and vehicle count.

4. Press `ESC` to exit the video window.

---

## Notes

* Works best with stable camera views and moderate traffic density.
* You can expand the project to detect other objects or use multiple counting lines.
* Optimizations such as using GPU, half-precision, and lower image resolution can improve speed.
