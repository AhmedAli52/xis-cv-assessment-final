# XIS AI / Computer Vision Technical Assessment

## Project Overview

This repository contains the implementation of the XIS AI / Computer Vision Technical Assessment. The project demonstrates an end-to-end computer vision pipeline for object segmentation and real-world measurement using camera calibration and deep learning.

The selected object for this assessment is a **book**. The pipeline performs camera calibration, dataset preparation, instance segmentation using Mask R-CNN, and pixel-to-millimeter measurement for estimating the real-world dimensions of the detected object.

---

## Objectives

The project implements the following tasks:

- Camera calibration using a checkerboard pattern
- Lens distortion correction using intrinsic calibration parameters
- Custom dataset collection and annotation
- Instance segmentation using Mask R-CNN
- Model evaluation on a held-out test dataset
- Pixel-to-millimeter conversion
- Real-world width and height estimation
- Measurement accuracy evaluation

---

## Project Features

- Intrinsic camera calibration
- Image undistortion
- COCO formatted dataset
- Mask R-CNN instance segmentation
- Automated inference pipeline
- Pixel-to-mm measurement pipeline
- IoU and Dice evaluation
- Measurement error analysis
- Modular project structure
- Complete technical documentation

---

# Repository Structure

```text
xis-cv-assessment-final/

├── calibration/
│   ├── calibration_images/
│   ├── output/
│   └── calibrate_camera.py
│
├── dataset/
│   ├── raw_images/
│   └── labeled_dataset/
│
├── docs/
│   ├── CALIBRATION_REPORT.md
│   ├── DATASET_CARD.md
│   ├── TRAINING_REPORT.md
│   ├── MEASUREMENT_REPORT.md
│   └── SETUP.md
│
├── inference/
│   ├── infer.py
│   ├── sample_inputs/
│   └── sample_outputs/
│
├── measurement/
│   ├── measure_book.py
│   └── results/
│
├── models/
│   └── torchvision_book/
│
├── README.md
└── requirements.txt
```

---

# Pipeline Workflow

```
Calibration Images
        │
        ▼
Camera Calibration
        │
        ▼
Image Undistortion
        │
        ▼
Dataset Collection
        │
        ▼
Image Annotation
        │
        ▼
Mask R-CNN Training
        │
        ▼
Inference
        │
        ▼
Object Segmentation
        │
        ▼
Pixel-to-mm Conversion
        │
        ▼
Width & Height Measurement
```

---

# Camera Calibration

The camera was calibrated using multiple checkerboard images captured from different viewpoints.

The calibration process generates:

- Camera intrinsic matrix
- Distortion coefficients
- Reprojection error
- Undistorted sample images

All calibration files are located in:

```
calibration/
```

---

# Dataset

The project uses a custom dataset consisting of images of a physical book.

Dataset includes:

- Raw images
- COCO annotations
- Train / Validation / Test split

Directory:

```
dataset/
```

---

# Model Training

The segmentation model is implemented using **Mask R-CNN with a ResNet-50 FPN backbone**.

Training includes:

- Custom dataset loader
- Model training
- Evaluation
- Model checkpoint generation

Model files are located in:

```
models/torchvision_book/
```

---

# Inference

A standalone inference pipeline is provided.

The inference module:

- Loads the trained model
- Performs object segmentation
- Generates annotated prediction images
- Saves results automatically

Run:

```bash
python inference/infer.py
```

Outputs are saved in:

```
inference/sample_outputs/
```

---

# Measurement Pipeline

The measurement module estimates the real-world dimensions of the segmented object.

The pipeline:

- Loads segmentation predictions
- Computes object dimensions in pixels
- Converts pixel measurements into millimeters
- Evaluates measurement accuracy

Run:

```bash
python measurement/measure_book.py
```

Results are saved in:

```
measurement/results/
```

---

# Results Summary

The project provides:

- Segmentation predictions
- IoU evaluation
- Dice score
- Width estimation
- Height estimation
- Pixel error
- Millimeter error

Detailed evaluation results are available in the generated measurement report.

---

# Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

Camera Calibration

```bash
python calibration/calibrate_camera.py
```

Inference

```bash
python inference/infer.py
```

Measurement

```bash
python measurement/measure_book.py
```

---

# Documentation

Additional project documentation is available in the `docs/` directory.

- Calibration Report
- Dataset Card
- Training Report
- Measurement Report
- Setup Guide

---

# Future Improvements

Possible future enhancements include:

- GPU inference optimization
- Real-time camera integration
- Automatic reference object detection
- Improved measurement accuracy
- Deployment as a web service

---

# Author


Prepared as part of the **XIS AI / Computer Vision Technical Assessment**.