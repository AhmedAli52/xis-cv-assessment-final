# Training Report

## Model
The project uses **Mask R-CNN** from **Torchvision** for instance segmentation.

## Task
The goal is to segment a single object class:
- **book**

## Dataset
Training was performed on the labeled COCO segmentation dataset prepared from Roboflow export.

Dataset split:
- Train: 54 images
- Validation: 15 images
- Test: 8 images

## Training Approach
The training pipeline loads the COCO-format segmentation dataset and fine-tunes a pretrained Mask R-CNN model for a single foreground class (`book`).

## Why Mask R-CNN
Mask R-CNN is suitable because it predicts:
- bounding boxes
- class labels
- segmentation masks

This makes it appropriate for both segmentation and later measurement of the detected object.

## Notes
This commit contains the training pipeline only. Model evaluation, inference, and measurement are documented separately.