# Setup Guide

# Overview

This document describes how to set up and run the XIS AI / Computer Vision Technical Assessment project.

The project implements an end-to-end computer vision pipeline consisting of:

- Camera calibration
- Dataset preparation
- Instance segmentation using Mask R-CNN
- Image inference
- Pixel-to-millimeter measurement

---

# Requirements

Software requirements:

- Python 3.10 or later
- Git
- pip

Operating System:

- Windows 10/11 (tested)
- Linux (compatible)

---

# Clone the Repository

```bash
git clone <repository-url>
cd xis-cv-assessment-final
```

---

# Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

# Repository Structure

```text
calibration/
dataset/
docs/
inference/
measurement/
models/
README.md
requirements.txt
```

---

# Camera Calibration

Run the calibration script:

```bash
python calibration/calibrate_camera.py
```

Generated files:

```
camera_calibration.npz
calibration_results.txt
undistorted_sample.jpeg
```

These files are stored in:

```
calibration/output/
```

---

# Dataset

The dataset is already organized into:

```
train/
valid/
test/
```

using the COCO Segmentation format.

---

# Model Training

Train the segmentation model:

```bash
python models/torchvision_book/train_mask_rcnn.py
```

The trained model is saved as:

```
models/torchvision_book/outputs/best_mask_rcnn_book.pth
```

---

# Inference

Run the inference pipeline:

```bash
python inference/infer.py
```

Input images:

```
inference/sample_inputs/
```

Output predictions:

```
inference/sample_outputs/
```

---

# Measurement

Run the measurement pipeline:

```bash
python measurement/measure_book.py
```

Results are generated in:

```
measurement/results/
```

---

# Documentation

Additional reports are available in the `docs/` directory:

- CALIBRATION_REPORT.md
- DATASET_CARD.md
- TRAINING_REPORT.md
- MEASUREMENT_REPORT.md

---

# Notes

- Camera calibration should be performed before measurement.
- The trained model must exist before running inference or measurement.
- The measurement pipeline uses the generated segmentation masks to estimate the physical dimensions of the selected object.

---

# Troubleshooting

## Missing Python Package

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model File Not Found

Ensure the trained model is available at:

```
models/torchvision_book/outputs/best_mask_rcnn_book.pth
```

## Calibration Files Missing

Run:

```bash
python calibration/calibrate_camera.py
```

before performing measurement.

---

# Conclusion

Following the steps in this guide will reproduce the complete computer vision pipeline from camera calibration through segmentation and final real-world measurement.