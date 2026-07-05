# Model Training Report

# Overview

This report describes the training procedure for the instance segmentation model developed for the XIS AI / Computer Vision Technical Assessment.

The objective of the model is to accurately segment a physical book from an image so that its dimensions can later be estimated in real-world units.

---

# Model Selection

The project uses **Mask R-CNN** implemented through the Torchvision library.

## Why Mask R-CNN?

Mask R-CNN was selected because it provides:

- Object detection
- Instance segmentation
- Pixel-level masks
- High localization accuracy
- Reliable object boundaries for measurement tasks

These capabilities make it suitable for pixel-to-millimeter estimation.

---

# Dataset

Training was performed using the custom annotated COCO segmentation dataset.

| Split | Images |
|-------|-------:|
| Training | 54 |
| Validation | 15 |
| Test | 8 |

Object Class:

```
book
```

Annotation format:

```
COCO Segmentation
```

---

# Training Pipeline

The training workflow consists of:

1. Loading the COCO dataset.
2. Creating custom PyTorch datasets.
3. Initializing Mask R-CNN.
4. Replacing the classification and mask heads.
5. Fine-tuning the model.
6. Saving the best-performing weights.

---

# Model Architecture

Framework:

```
Torchvision
```

Model:

```
Mask R-CNN
```

Backbone:

```
ResNet-50 FPN
```

Number of Classes:

```
2
```

- Background
- Book

---

# Training Configuration

The training pipeline includes:

- Custom dataset loader
- COCO annotations
- Fine-tuning of the pretrained backbone
- Model checkpoint generation
- Evaluation on a held-out test set

The complete implementation is available in:

```
models/torchvision_book/
```

---

# Evaluation

The trained Mask R-CNN model was evaluated on the held-out test dataset.

Evaluation Results

| Metric | Value |
|--------|------:|
| Test Images | 8 |
| Successful Detections | 8 |
| Detection Rate | 100% |
| Mean IoU | 0.8844 |
| Mean Dice Score | 0.9374 |

The evaluation demonstrates strong segmentation performance, with a mean IoU of 0.8844 and a mean Dice score of 0.9374 across the complete test split.

---

# Generated Outputs

Training produces:

- Trained model weights

```
best_mask_rcnn_book.pth
```

Additional outputs include:

- Segmentation predictions
- Measurement reports
- Evaluation metrics

---

# Strengths

The selected architecture provides:

- Accurate object segmentation
- Reliable mask generation
- Pixel-level localization
- Strong compatibility with measurement applications

---

# Limitations

Current limitations include:

- Single object category
- Relatively small dataset
- CPU-based execution during testing
- Limited evaluation dataset

---

# Future Improvements

Potential improvements include:

- Larger training dataset
- Data augmentation
- Hyperparameter optimization
- GPU-based training and inference
- Multiple object classes
- Real-time deployment

---

# Conclusion

Mask R-CNN successfully provides accurate instance segmentation for the selected object and forms the foundation for the subsequent pixel-to-millimeter measurement pipeline. The trained model is integrated into both the inference and measurement modules of this project.