# Measurement Report

# Overview

This report describes the pixel-to-millimeter measurement pipeline developed for the XIS AI / Computer Vision Technical Assessment.

The objective of the measurement stage is to estimate the real-world dimensions of a segmented object by combining camera calibration with instance segmentation.

---

# Measurement Methodology

The measurement pipeline consists of the following steps:

1. Load the calibrated image.
2. Perform object segmentation using the trained Mask R-CNN model.
3. Extract the predicted segmentation mask.
4. Compute the object dimensions in pixels using the minimum-area bounding rectangle.
5. Convert pixel measurements into millimeters.
6. Compare predicted measurements against the ground truth.

---

# Real Object Dimensions

The selected object is a book.

Reference dimensions used during evaluation:

| Dimension | Value |
|----------|-------:|
| Width | 135 mm |
| Height | 195 mm |

---

# Measurement Process

The implementation uses:

- OpenCV
- Mask R-CNN
- COCO annotations
- Minimum Area Rectangle (`cv2.minAreaRect()`)

The segmentation mask is converted into pixel dimensions, which are then transformed into real-world measurements using the known object dimensions.

---

# Evaluation Results

The evaluation was performed on the complete test split.

| Metric | Value |
|--------|------:|
| Images Evaluated | 8 |
| Mean IoU | 0.8844 |
| Mean Dice Score | 0.9374 |
| Mean Width Error | 45.58 px / 8.74 mm |
| Mean Height Error | 56.64 px / 11.17 mm |

These values were generated using the measurement pipeline included in this repository.

---

# Generated Output

Running

```bash
python measurement/measure_book.py
```

produces:

```text
measurement/results/
└── measurement_results.txt
```

which contains detailed per-image measurements and summary statistics.

---

# Discussion

The evaluation demonstrates that the trained Mask R-CNN model is capable of accurately segmenting the selected object and estimating its physical dimensions.

The segmentation quality, reflected by the IoU and Dice scores, supports reliable pixel-based measurement. Remaining measurement errors are primarily influenced by segmentation accuracy, object orientation, and image perspective.

---

# Limitations

Current limitations include:

- Evaluation performed on a limited test set.
- Single object category.
- Measurements depend on accurate segmentation.
- Perspective variations can influence estimated dimensions.

---

# Future Improvements

Possible improvements include:

- Perspective correction.
- Automatic reference object detection.
- Multi-object measurement.
- GPU-accelerated inference.
- Evaluation on a larger dataset.

---

# Conclusion

The measurement pipeline successfully converts segmentation results into real-world object dimensions. Combined with camera calibration and Mask R-CNN segmentation, it provides a complete end-to-end workflow for estimating physical measurements from images.