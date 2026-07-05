# Camera Calibration Report

# Objective

The objective of camera calibration is to estimate the intrinsic camera parameters and lens distortion coefficients required for accurate image rectification. Since lens distortion affects the apparent size and shape of objects, all images used for segmentation and measurement are undistorted before further processing.

---

# Calibration Target

A printed checkerboard pattern was used for intrinsic calibration.

| Parameter | Value |
|-----------|------|
| Checkerboard Inner Corners | 10 × 6 |
| Square Size | 20 mm |

---

# Calibration Dataset

A total of **27 calibration images** were captured from different viewpoints and distances.

| Description | Count |
|------------|------:|
| Total Images | 27 |
| Successful Corner Detections | 11 |
| Rejected Images | 16 |

Only successful detections were used to estimate the intrinsic camera parameters.

---

# Calibration Methodology

The calibration process was implemented using OpenCV.

The following functions were used:

- `cv2.findChessboardCorners()`
- `cv2.cornerSubPix()`
- `cv2.calibrateCamera()`
- `cv2.undistort()`

The detected checkerboard corners were refined to sub-pixel accuracy before estimating the camera matrix and distortion coefficients.

---

# Calibration Results

The calibration process produced:

- Camera intrinsic matrix
- Distortion coefficients
- Mean reprojection error
- Undistortion parameters

Mean Reprojection Error

```
0.3935 pixels
```

A reprojection error below **0.5 pixels** indicates a reliable calibration suitable for measurement applications.

---

# Generated Files

The calibration module generates:

```
camera_calibration.npz
calibration_results.txt
undistorted_sample.jpeg
```

These files are stored inside:

```
calibration/output/
```

---

# Calibration Usage

The computed intrinsic parameters are used throughout the project.

Before segmentation or measurement:

1. Load calibration parameters.
2. Undistort the captured image.
3. Perform object segmentation.
4. Measure object dimensions.

This ensures that real-world measurements are not affected by lens distortion.

---

# Limitations

Calibration accuracy depends on:

- Number of successful checkerboard detections
- Checkerboard visibility
- Image quality
- Lighting conditions
- Camera stability during capture

Increasing the number of successful calibration images can further improve measurement accuracy.

---

# Conclusion

The calibration process successfully estimated the intrinsic camera parameters with a mean reprojection error of **0.3935 pixels**. The resulting calibration parameters are used throughout the segmentation and measurement pipeline to improve the accuracy of real-world dimension estimation.