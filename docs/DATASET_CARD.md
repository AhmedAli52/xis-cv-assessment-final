# Dataset Card

# Dataset Overview

This dataset was created for the XIS AI / Computer Vision Technical Assessment. The objective of the dataset is to train and evaluate an instance segmentation model capable of accurately segmenting a physical book and supporting real-world dimension measurement.

---

# Object of Interest

The selected object is a **book**.

The book was selected because:

- It has well-defined boundaries.
- It is easy to capture from different viewpoints.
- It provides reliable dimensions for measurement validation.
- It is suitable for segmentation-based measurement tasks.

---

# Data Collection

Images were captured manually using a mobile phone camera.

To improve dataset diversity, images were collected with varying:

- Camera distances
- Viewing angles
- Object orientations
- Positions within the image frame
- Lighting conditions

This variation improves the robustness and generalization capability of the segmentation model.

---

# Dataset Statistics

| Property | Value |
|----------|------:|
| Total Images | 77 |
| Object Class | Book |
| Annotation Type | Polygon Segmentation |
| Export Format | COCO Segmentation |

---

# Dataset Split

| Split | Images |
|-------|-------:|
| Training | 54 |
| Validation | 15 |
| Test | 8 |

---

# Annotation

The dataset was annotated using **Roboflow**.

Annotation details:

- Instance Segmentation
- Polygon Masks
- Single Object Class
- COCO Segmentation Export

Class Label

```
book
```

---

# Dataset Structure

```text
dataset/

├── raw_images/
├── labeled_dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── README.dataset.txt
│   └── README.roboflow.txt
│
└── rename_images.py
```

---

# Intended Use

The dataset is intended for:

- Camera calibration validation
- Instance segmentation
- Model evaluation
- Pixel-to-millimeter measurement
- Computer vision experimentation

---

# Data Quality

The dataset includes images captured under varying conditions to improve model robustness.

Images contain variation in:

- Scale
- Rotation
- Perspective
- Background
- Illumination

The dataset was manually reviewed before annotation to remove unusable samples.

---

# Limitations

Current limitations include:

- Single object category
- Moderate dataset size
- Mobile camera acquisition only
- Indoor image collection

Future work may include:

- Additional object categories
- Outdoor environments
- Larger dataset
- Multiple camera devices

---

# Conclusion

The dataset provides a structured collection of annotated book images suitable for training, evaluating, and validating an instance segmentation model as part of the complete computer vision measurement pipeline.