# Dataset Card

## Object of Interest
The chosen object for this project is a **book**.

## Raw Data Collection
Images of the book were captured manually using a phone camera from different:
- distances
- viewing angles
- orientations
- positions in the frame

This was done to create variation for robust segmentation training.

## Dataset Size
- Total collected raw images: **77**

## Annotation
The dataset was annotated using **Roboflow** for **instance segmentation**.

- Class label used: **book**
- Annotation type: **polygon segmentation**
- Export format: **COCO Segmentation**

## Dataset Split
The labeled dataset was split as follows:

- **Train:** 54 images
- **Validation:** 15 images
- **Test:** 8 images

## Folder Structure
```text
dataset/
├── raw_images/
├── labeled_dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── README.dataset.txt
│   └── README.roboflow.txt
└── rename_images.py