import cv2
import numpy as np
import glob
import os
import sys

# ----------------------------------------------------------
# Disable OpenCL (avoids unnecessary OpenCL warnings)
# ----------------------------------------------------------

cv2.ocl.setUseOpenCL(False)

# ----------------------------------------------------------
# Checkerboard Settings
# ----------------------------------------------------------

# 11 x 7 squares  -> 10 x 6 inner corners
CHECKERBOARD = (10, 6)

# Checkerboard square size (mm)
SQUARE_SIZE = 20.0

IMAGE_DIR = "calibration_images"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ----------------------------------------------------------
# Prepare object points
# ----------------------------------------------------------

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# ----------------------------------------------------------
# Read Images
# ----------------------------------------------------------

images = sorted(set(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
))

print("=" * 60)
print(f"Found {len(images)} calibration images")
print("=" * 60)

if len(images) == 0:
    print("No calibration images found.")
    sys.exit()

successful = 0
failed = 0

gray = None

# ----------------------------------------------------------
# Detect Checkerboard
# ----------------------------------------------------------

for image_path in images:

    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Cannot read {image_path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_FILTER_QUADS
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags
    )

    if found:

        successful += 1

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners)

        vis = image.copy()

        cv2.drawChessboardCorners(
            vis,
            CHECKERBOARD,
            corners,
            found
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                os.path.basename(image_path)
            ),
            vis
        )

        print(f"[OK]     {os.path.basename(image_path)}")

    else:

        failed += 1
        print(f"[FAILED] {os.path.basename(image_path)}")

# ----------------------------------------------------------
# Check Enough Images
# ----------------------------------------------------------

print("\n")
print("=" * 60)
print("Detection Summary")
print("=" * 60)

print(f"Successful Images : {successful}")
print(f"Rejected Images   : {failed}")

if successful < 10:
    print("\nERROR: Not enough checkerboards detected.")
    print("Capture more images or verify checkerboard size.")
    sys.exit()

# ----------------------------------------------------------
# Calibration
# ----------------------------------------------------------

print("\nRunning Camera Calibration...\n")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

# ----------------------------------------------------------
# Reprojection Error
# ----------------------------------------------------------

total_error = 0

for i in range(len(objpoints)):

    projected_points, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv2.norm(
        imgpoints[i],
        projected_points,
        cv2.NORM_L2
    ) / len(projected_points)

    total_error += error

mean_error = total_error / len(objpoints)

# ----------------------------------------------------------
# Print Results
# ----------------------------------------------------------

print("=" * 60)
print("Calibration Results")
print("=" * 60)

print("\nCamera Matrix:\n")
print(camera_matrix)

print("\nDistortion Coefficients:\n")
print(dist_coeffs)

print(f"\nMean Reprojection Error : {mean_error:.4f} pixels")

# ----------------------------------------------------------
# Save Calibration
# ----------------------------------------------------------

np.savez(
    os.path.join(
        OUTPUT_DIR,
        "camera_calibration.npz"
    ),
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    image_width=gray.shape[1],
    image_height=gray.shape[0],
    checkerboard=CHECKERBOARD,
    square_size=SQUARE_SIZE,
    reprojection_error=mean_error
)

# ----------------------------------------------------------
# Save Report
# ----------------------------------------------------------

with open(
    os.path.join(
        OUTPUT_DIR,
        "calibration_results.txt"
    ),
    "w"
) as f:

    f.write("CAMERA CALIBRATION REPORT\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Successful Images : {successful}\n")
    f.write(f"Rejected Images   : {failed}\n\n")

    f.write("Camera Matrix\n")
    f.write(str(camera_matrix))
    f.write("\n\n")

    f.write("Distortion Coefficients\n")
    f.write(str(dist_coeffs))
    f.write("\n\n")

    f.write(
        f"Mean Reprojection Error : {mean_error:.6f} pixels\n"
    )

# ----------------------------------------------------------
# Save Undistorted Sample
# ----------------------------------------------------------

sample = cv2.imread(images[0])

h, w = sample.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    1,
    (w, h)
)

undistorted = cv2.undistort(
    sample,
    camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "undistorted_sample.jpeg"
    ),
    undistorted
)

# ----------------------------------------------------------
# Finished
# ----------------------------------------------------------

print("\nCalibration completed successfully.\n")

print("Files Generated")
print("---------------------------")
print("camera_calibration.npz")
print("calibration_results.txt")
print("undistorted_sample.jpeg")
print("checkerboard visualization images")
print("---------------------------")