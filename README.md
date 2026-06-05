# MultiTracking: Multi-Vehicle Tracking System

MultiTracking is a web-based multi-vehicle tracking system built with YOLOv5, DeepSORT, and a custom DeepSORT-style tracker. The project supports vehicle detection, multi-object tracking, video visualization, dataset evaluation, upload analysis, webcam tracking, and tracker comparison.

The system provides two tracking options:

1. **Original DeepSORT**
   Baseline tracker using the `deep-sort-realtime` library and a pretrained Re-ID embedder.

2. **Custom DeepSORT**
   A self-implemented tracker that includes Kalman Filter, IoU Matching, Appearance Matching, Hungarian Assignment, and Track Lifecycle management.

The project is designed for studying, evaluating, and comparing default DeepSORT with a custom implementation in a multi-vehicle tracking task.

---

## Dataset

The vehicle tracking dataset can be downloaded from:

[Vehicle Tracking Dataset](https://zenodo.org/records/18195750)

After downloading, place the dataset inside:

```text
data/vehicle_tracking/
```

Expected dataset structure:

```text
data/vehicle_tracking/
├── AICC22-Custom
│   ├── AICC22-Custom_GroundTruth.json
│   ├── AICC22-Custom_GroundTruth.txt
│   ├── AICC22-Custom_GroundTruth-video.mp4
│   └── AICC22-Custom_Original-video.mp4
└── VNTraffic
    ├── VNTraffic_GroundTruth.json
    ├── VNTraffic_GroundTruth.txt
    ├── VNTraffic_GroundTruth-video.mp4
    └── VNTraffic_Original-video.mp4
```

---

## Project Structure

```text
MultiTracking/
├── app.py
├── configs/
│   └── config.yaml
├── data/
│   └── vehicle_tracking/
├── models/
│   ├── ckpt.t7
│   ├── yolov5n.pt
│   └── yolov5nu.pt
├── results/
│   ├── compare/
│   ├── dataset/
│   ├── upload/
│   └── webcam/
├── scripts/
│   └── compare_trackers.py
├── src/
│   ├── custom_deepsort/
│   │   ├── appearance.py
│   │   ├── detection.py
│   │   ├── iou_matching.py
│   │   ├── kalman_filter.py
│   │   ├── linear_assignment.py
│   │   ├── track.py
│   │   └── tracker.py
│   ├── compare.py
│   ├── dataset.py
│   ├── detector.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── tracker.py
│   ├── utils.py
│   ├── video_processor.py
│   ├── visualizer.py
│   └── webcam_processor.py
├── static/
├── templates/
├── uploads/
├── requirements.txt
└── README.md
```

---

## Main Features

### 1. Dataset Mode

Run vehicle tracking on prepared datasets with ground truth annotations.

This mode computes evaluation metrics such as:

* Precision
* Recall
* F1-score
* MOTA proxy
* MOTP proxy
* ID Switches proxy
* Processing FPS
* Average detector time
* Average tracker time

### 2. Upload Mode

Upload any video and run vehicle tracking without ground truth.

This mode reports tracking statistics such as:

* Number of unique tracked vehicles
* Average vehicles per frame
* Maximum vehicles per frame
* Track length statistics
* Processing FPS
* Detector and tracker runtime

### 3. Webcam Mode

Run real-time tracking using a webcam.

This mode supports:

* Live vehicle detection and tracking
* FPS display
* Active track count
* Recording webcam tracking results
* Saving video, TXT tracking logs, and metrics

### 4. Compare Mode

Compare the original DeepSORT tracker with the Custom DeepSORT V2 tracker on the same video or dataset.

The comparison includes:

* Processing FPS
* Average tracker time
* Unique tracks
* Average objects per frame
* Precision
* Recall
* F1-score
* MOTA proxy
* ID Switches proxy

---

## Models

The project uses the following model files:

```text
models/yolov5n.pt
models/yolov5nu.pt
models/ckpt.t7
```

Default detector:

```text
models/yolov5n.pt
```

The YOLO model is used for vehicle detection. The DeepSORT baseline uses a pretrained Re-ID embedder through the `deep-sort-realtime` package.

---

## Vehicle Classes

The detector focuses on vehicle classes from the COCO dataset:

```text
2 -> car
3 -> motorcycle
5 -> bus
7 -> truck
```

---

## Tracking Methods

### Original DeepSORT

The original DeepSORT baseline uses:

```text
YOLO detector
OpenCV
NumPy
PyTorch
scipy Hungarian Assignment
deep-sort-realtime
Pretrained Re-ID model
```

This tracker is used as the baseline for comparison.

### Custom DeepSORT V2

The custom tracker is implemented inside:

```text
src/custom_deepsort/
```

It includes the following components:

```text
Kalman Filter
IoU Matching
Appearance Matching
Hungarian Assignment
Track Lifecycle
```

The optimized Custom DeepSORT V2 uses:

* Kalman Filter for motion prediction
* HSV + LAB histogram features for appearance representation
* IoU distance for bounding box overlap matching
* Motion distance from Kalman prediction
* Combined cost matching
* Hungarian Assignment for optimal track-detection association
* Track states: Tentative, Confirmed, Deleted

The custom tracker is designed to be lightweight and explainable while still maintaining competitive tracking quality.

---

## Installation

### 1. Clone or open the project

```bash
cd MultiTracking
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Upgrade pip, setuptools, and wheel

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Web Application

Start the Flask application:

```bash
python app.py
```

Then open the website in your browser:

```text
http://127.0.0.1:5000
```

If you want to access the app from another device on the same Wi-Fi or LAN network, use the LAN address shown in the terminal, for example:

```text
http://192.168.x.x:5000
```

---

## Running Tracker Comparison from Terminal

Compare both trackers on the VNTraffic dataset:

```bash
python scripts/compare_trackers.py --dataset VNTraffic
```

Compare both trackers on the AICC22-Custom dataset:

```bash
python scripts/compare_trackers.py --dataset AICC22-Custom
```

Run a quick test with only the first 100 frames:

```bash
python scripts/compare_trackers.py --dataset VNTraffic --max-frames 100
```

Run comparison on a custom video:

```bash
python scripts/compare_trackers.py --video uploads/sample.mp4 --name sample_video
```

---

## Output Files

The system saves output files into the `results/` directory.

### Dataset results

```text
results/dataset/
├── videos/
├── txt/
└── metrics/
```

### Upload results

```text
results/upload/
├── videos/
├── txt/
└── metrics/
```

### Webcam results

```text
results/webcam/
├── videos/
├── txt/
└── metrics/
```

### Tracker comparison results

```text
results/compare/
├── videos/
├── txt/
└── metrics/
```

Each run can generate:

```text
tracking video: .mp4
tracking log: .txt
evaluation metrics: .json
comparison summary: .json
```

---

## Tracking TXT Format

Tracking results are saved in CSV-style TXT files with the following format:

```text
frame,id,x,y,w,h,conf,class,visibility
```

Field meanings:

| Field      | Description                                 |
| ---------- | ------------------------------------------- |
| frame      | Frame index                                 |
| id         | Tracking ID                                 |
| x          | Top-left x coordinate                       |
| y          | Top-left y coordinate                       |
| w          | Bounding box width                          |
| h          | Bounding box height                         |
| conf       | Detection confidence                        |
| class      | Vehicle class ID                            |
| visibility | 1 if matched with detection, 0 if predicted |

---

## Evaluation Metrics

For datasets with ground truth, the system computes proxy tracking metrics based on IoU matching.

| Metric            | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| Precision         | Correct predictions over all predictions                    |
| Recall            | Correct predictions over all ground truth objects           |
| F1-score          | Harmonic mean of precision and recall                       |
| MOTP proxy        | Average IoU of matched boxes                                |
| MOTA proxy        | Approximate tracking accuracy using FP, FN, and ID switches |
| ID Switches proxy | Number of estimated identity switches                       |
| Processing FPS    | End-to-end video processing speed                           |
| Tracker Time      | Average tracker runtime per frame                           |

---

## Final Full-Dataset Comparison Results

### VNTraffic

| Tracker            |     FPS | Tracker Time | Precision | Recall | F1-score | MOTA Proxy | ID Switches |
| ------------------ | ------: | -----------: | --------: | -----: | -------: | ---------: | ----------: |
| DeepSORT original  |  2.7941 |  160.0233 ms |    0.8319 | 0.2716 |   0.4096 |     0.2146 |          35 |
| Custom DeepSORT V2 | 14.5728 |    6.5093 ms |    0.9257 | 0.2604 |   0.4065 |     0.2381 |          23 |

### AICC22-Custom

| Tracker            |     FPS | Tracker Time | Precision | Recall | F1-score | MOTA Proxy | ID Switches |
| ------------------ | ------: | -----------: | --------: | -----: | -------: | ---------: | ----------: |
| DeepSORT original  |  4.9468 |   90.8557 ms |    0.5546 | 0.7030 |   0.6200 |     0.1342 |           5 |
| Custom DeepSORT V2 | 16.0393 |    5.0226 ms |    0.7005 | 0.6810 |   0.6906 |     0.3890 |           1 |

The results show that Custom DeepSORT V2 significantly improves processing speed and ID stability. It also improves MOTA proxy on both datasets, while maintaining competitive F1-score compared with the original DeepSORT baseline.

---

## Notes

* The current implementation is CPU-friendly.
* The default detector is YOLOv5n.
* The custom tracker does not use a deep Re-ID model.
* Appearance matching in Custom DeepSORT V2 is based on HSV and LAB histogram features.
* DeepSORT original is used as a baseline for comparison.
* Metrics are proxy metrics and are computed using IoU-based matching with ground truth.

---

## Deactivate the Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

---

## Summary

This project demonstrates a complete multi-vehicle tracking pipeline using YOLOv5 and DeepSORT. In addition to the original DeepSORT baseline, the project includes a custom DeepSORT-style tracker implemented from core tracking components. The comparison results show that the custom tracker achieves much faster processing speed and better ID stability, making it suitable for lightweight and explainable vehicle tracking applications.
