# Calibration Report

## Calibration Target
- Checkerboard inner corners: 10 × 6
- Square size: 20 mm

## Images
- Total calibration images: 27
- Successful detections: 11
- Rejected images: 16

## Method
Camera calibration was performed using OpenCV checkerboard calibration:
- `cv2.findChessboardCorners()`
- `cv2.cornerSubPix()`
- `cv2.calibrateCamera()`

## Results
- Mean reprojection error: 0.3935 pixels

## Output Files
- `camera_calibration.npz`
- `calibration_results.txt`
- `undistorted_sample.jpeg`

## Notes
The calibration parameters will be used to undistort measurement images before segmentation and size estimation.