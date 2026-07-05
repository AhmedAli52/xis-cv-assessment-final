import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ==========================================================
# Paths
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "labeled_dataset", "test")
TEST_JSON = os.path.join(TEST_DIR, "_annotations.coco.json")

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best_mask_rcnn_book.pth")
OUTPUT_REPORT = os.path.join(BASE_DIR, "outputs", "evaluation_results.txt")

# ==========================================================
# Config
# ==========================================================
NUM_CLASSES = 2   # background + book
SCORE_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Build model
# ==========================================================
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=None)

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

ious = []
dices = []
successful_detections = 0
total_images = 0

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
            print(f"[WARNING] Image not found: {image_name}")
            continue

        # Load image
        pil_image = Image.open(image_path).convert("RGB")
        image_np = np.array(pil_image)
        height, width = image_np.shape[:2]

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(DEVICE)

        # Get GT annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            continue

        # Merge all GT masks into one mask
        gt_mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:
            ann_mask = coco.annToMask(ann)
            gt_mask = np.maximum(gt_mask, ann_mask)

        # Predict
        outputs = model([image_tensor])[0]

        scores = outputs["scores"].detach().cpu().numpy()
        masks = outputs["masks"].detach().cpu().numpy()

        total_images += 1

        if len(scores) == 0:
            print(f"{image_name}: No prediction")
            continue

        valid_indices = np.where(scores >= SCORE_THRESHOLD)[0]

        if len(valid_indices) == 0:
            print(f"{image_name}: No prediction above threshold")
            continue

        # Keep only best prediction
        best_idx = valid_indices[np.argmax(scores[valid_indices])]
        pred_mask = masks[best_idx, 0] > 0.5

        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)

        ious.append(iou)
        dices.append(dice)
        successful_detections += 1

        print(f"{image_name} -> IoU: {iou:.4f}, Dice: {dice:.4f}")

# ==========================================================
# Final results
# ==========================================================
mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
mean_dice = np.mean(dices) if len(dices) > 0 else 0.0
detection_rate = successful_detections / total_images if total_images > 0 else 0.0

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Total test images         : {total_images}")
print(f"Successful detections     : {successful_detections}")
print(f"Detection rate            : {detection_rate:.4f}")
print(f"Mean IoU                  : {mean_iou:.4f}")
print(f"Mean Dice Score           : {mean_dice:.4f}")
print("=" * 60)

# Save report
with open(OUTPUT_REPORT, "w") as f:
    f.write("MODEL EVALUATION REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Total test images         : {total_images}\n")
    f.write(f"Successful detections     : {successful_detections}\n")
    f.write(f"Detection rate            : {detection_rate:.4f}\n")
    f.write(f"Mean IoU                  : {mean_iou:.4f}\n")
    f.write(f"Mean Dice Score           : {mean_dice:.4f}\n")

print(f"\nEvaluation report saved to: {OUTPUT_REPORT}")