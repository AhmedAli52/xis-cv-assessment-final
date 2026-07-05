import os
import cv2
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ==========================================================
# Paths
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "labeled_dataset", "test")
TEST_JSON = os.path.join(TEST_DIR, "_annotations.coco.json")

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "torchvision_book",
    "outputs",
    "best_mask_rcnn_book.pth",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
REPORT_PATH = os.path.join(OUTPUT_DIR, "measurement_results.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Real book dimensions (mm)
# ==========================================================
REAL_BOOK_WIDTH_MM = 135.0
REAL_BOOK_HEIGHT_MM = 195.0

# ==========================================================
# Config
# ==========================================================
NUM_CLASSES = 2
SCORE_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Build model
# ==========================================================
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


# ==========================================================
# Metrics
# ==========================================================
def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 0.0
    return (2.0 * intersection) / total


# ==========================================================
# Get width/height from mask using rotated rectangle
# ==========================================================
def get_mask_dimensions(mask):
    """
    mask: binary mask (H, W), values 0/1 or False/True
    returns: width_px, height_px
    """

    mask_uint8 = (mask.astype(np.uint8)) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(largest_contour)
    (center_x, center_y), (w, h), angle = rect

    # Ensure width = shorter side, height = longer side
    width_px = min(w, h)
    height_px = max(w, h)

    return width_px, height_px


# ==========================================================
# Convert pixel error to mm error using GT scale
# ==========================================================
def pixel_error_to_mm_error(pred_w_px, pred_h_px, gt_w_px, gt_h_px):
    """
    Convert prediction error from pixels to millimeters using GT mask scale.
    """

    # mm per pixel from GT object dimensions
    mm_per_px_width = REAL_BOOK_WIDTH_MM / gt_w_px if gt_w_px > 0 else 0
    mm_per_px_height = REAL_BOOK_HEIGHT_MM / gt_h_px if gt_h_px > 0 else 0

    width_error_px = abs(pred_w_px - gt_w_px)
    height_error_px = abs(pred_h_px - gt_h_px)

    width_error_mm = width_error_px * mm_per_px_width
    height_error_mm = height_error_px * mm_per_px_height

    pred_width_mm = pred_w_px * mm_per_px_width
    pred_height_mm = pred_h_px * mm_per_px_height

    return (
        width_error_px,
        height_error_px,
        width_error_mm,
        height_error_mm,
        pred_width_mm,
        pred_height_mm
    )


# ==========================================================
# Load model
# ==========================================================
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(f"Using device: {DEVICE}")
print(f"Loaded model from: {MODEL_PATH}")

# ==========================================================
# Load COCO annotations
# ==========================================================
coco = COCO(TEST_JSON)
image_ids = coco.getImgIds()

all_results = []

ious = []
dices = []

width_errors_px = []
height_errors_px = []

width_errors_mm = []
height_errors_mm = []

print(f"Found {len(image_ids)} test images")

# ==========================================================
# Evaluation loop
# ==========================================================
with torch.no_grad():
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        image_name = img_info["file_name"]
        image_path = os.path.join(TEST_DIR, image_name)

        if not os.path.exists(image_path):
            print(f"[WARNING] Missing image: {image_name}")
            continue

        # Load image
        pil_image = Image.open(image_path).convert("RGB")
        image_np = np.array(pil_image)
        h, w = image_np.shape[:2]

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(DEVICE)

        # ---------------- GT mask ----------------
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            print(f"[WARNING] No annotations for {image_name}")
            continue

        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            ann_mask = coco.annToMask(ann)
            gt_mask = np.maximum(gt_mask, ann_mask)

        gt_w_px, gt_h_px = get_mask_dimensions(gt_mask)

        if gt_w_px is None or gt_h_px is None:
            print(f"[WARNING] Could not compute GT dimensions for {image_name}")
            continue

        # ---------------- Prediction ----------------
        outputs = model([image_tensor])[0]
        scores = outputs["scores"].detach().cpu().numpy()
        masks = outputs["masks"].detach().cpu().numpy()

        if len(scores) == 0:
            print(f"[WARNING] No prediction for {image_name}")
            continue

        valid_indices = np.where(scores >= SCORE_THRESHOLD)[0]
        if len(valid_indices) == 0:
            print(f"[WARNING] No prediction above threshold for {image_name}")
            continue

        best_idx = valid_indices[np.argmax(scores[valid_indices])]
        pred_mask = masks[best_idx, 0] > 0.5

        pred_w_px, pred_h_px = get_mask_dimensions(pred_mask)

        if pred_w_px is None or pred_h_px is None:
            print(f"[WARNING] Could not compute predicted dimensions for {image_name}")
            continue

        # ---------------- Metrics ----------------
        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)

        (
            width_error_px,
            height_error_px,
            width_error_mm,
            height_error_mm,
            pred_width_mm,
            pred_height_mm
        ) = pixel_error_to_mm_error(
            pred_w_px, pred_h_px,
            gt_w_px, gt_h_px
        )

        ious.append(iou)
        dices.append(dice)

        width_errors_px.append(width_error_px)
        height_errors_px.append(height_error_px)

        width_errors_mm.append(width_error_mm)
        height_errors_mm.append(height_error_mm)

        result = {
            "image_name": image_name,
            "gt_width_px": gt_w_px,
            "gt_height_px": gt_h_px,
            "pred_width_px": pred_w_px,
            "pred_height_px": pred_h_px,
            "pred_width_mm": pred_width_mm,
            "pred_height_mm": pred_height_mm,
            "width_error_px": width_error_px,
            "height_error_px": height_error_px,
            "width_error_mm": width_error_mm,
            "height_error_mm": height_error_mm,
            "iou": iou,
            "dice": dice
        }

        all_results.append(result)

        print(f"\n{image_name}")
        print(f"  GT size (px)       : {gt_w_px:.2f} x {gt_h_px:.2f}")
        print(f"  Pred size (px)     : {pred_w_px:.2f} x {pred_h_px:.2f}")
        print(f"  Pred size (mm)     : {pred_width_mm:.2f} x {pred_height_mm:.2f}")
        print(f"  Width error        : {width_error_px:.2f} px / {width_error_mm:.2f} mm")
        print(f"  Height error       : {height_error_px:.2f} px / {height_error_mm:.2f} mm")
        print(f"  IoU                : {iou:.4f}")
        print(f"  Dice               : {dice:.4f}")

# ==========================================================
# Final summary
# ==========================================================
mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
mean_dice = np.mean(dices) if len(dices) > 0 else 0.0

mean_width_error_px = np.mean(width_errors_px) if len(width_errors_px) > 0 else 0.0
mean_height_error_px = np.mean(height_errors_px) if len(height_errors_px) > 0 else 0.0

mean_width_error_mm = np.mean(width_errors_mm) if len(width_errors_mm) > 0 else 0.0
mean_height_error_mm = np.mean(height_errors_mm) if len(height_errors_mm) > 0 else 0.0

print("\n" + "=" * 70)
print("MEASUREMENT ACCURACY RESULTS")
print("=" * 70)
print(f"Images evaluated              : {len(all_results)}")
print(f"Mean IoU                      : {mean_iou:.4f}")
print(f"Mean Dice                     : {mean_dice:.4f}")
print(f"Mean width error             : {mean_width_error_px:.2f} px / {mean_width_error_mm:.2f} mm")
print(f"Mean height error            : {mean_height_error_px:.2f} px / {mean_height_error_mm:.2f} mm")
print("=" * 70)

# ==========================================================
# Save report
# ==========================================================
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("BOOK MEASUREMENT ACCURACY REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Real book width  : {REAL_BOOK_WIDTH_MM} mm\n")
    f.write(f"Real book height : {REAL_BOOK_HEIGHT_MM} mm\n\n")

    for r in all_results:
        f.write(f"Image: {r['image_name']}\n")
        f.write(f"  GT size (px)       : {r['gt_width_px']:.2f} x {r['gt_height_px']:.2f}\n")
        f.write(f"  Pred size (px)     : {r['pred_width_px']:.2f} x {r['pred_height_px']:.2f}\n")
        f.write(f"  Pred size (mm)     : {r['pred_width_mm']:.2f} x {r['pred_height_mm']:.2f}\n")
        f.write(f"  Width error        : {r['width_error_px']:.2f} px / {r['width_error_mm']:.2f} mm\n")
        f.write(f"  Height error       : {r['height_error_px']:.2f} px / {r['height_error_mm']:.2f} mm\n")
        f.write(f"  IoU                : {r['iou']:.4f}\n")
        f.write(f"  Dice               : {r['dice']:.4f}\n")
        f.write("\n")

    f.write("=" * 70 + "\n")
    f.write("FINAL SUMMARY\n")
    f.write("=" * 70 + "\n")
    f.write(f"Images evaluated              : {len(all_results)}\n")
    f.write(f"Mean IoU                      : {mean_iou:.4f}\n")
    f.write(f"Mean Dice                     : {mean_dice:.4f}\n")
    f.write(f"Mean width error             : {mean_width_error_px:.2f} px / {mean_width_error_mm:.2f} mm\n")
    f.write(f"Mean height error            : {mean_height_error_px:.2f} px / {mean_height_error_mm:.2f} mm\n")

print(f"\nMeasurement report saved to: {REPORT_PATH}")