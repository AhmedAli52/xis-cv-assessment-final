import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ==========================================================
# Paths
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

TEST_DIR = os.path.join(BASE_DIR, "sample_inputs")
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "torchvision_book",
    "outputs",
    "best_mask_rcnn_book.pth",
)

OUTPUT_DIR = os.path.join(BASE_DIR, "sample_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )
    # Replace box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


# ==========================================================
# Draw only ONE best prediction
# ==========================================================
def draw_best_prediction(image_bgr, outputs, threshold=0.5):
    boxes = outputs["boxes"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu().numpy()
    masks = outputs["masks"].detach().cpu().numpy()

    if len(scores) == 0:
        return image_bgr, False

    # Keep only predictions above threshold
    valid_indices = np.where(scores >= threshold)[0]

    if len(valid_indices) == 0:
        return image_bgr, False

    # Select highest-score prediction only
    best_idx = valid_indices[np.argmax(scores[valid_indices])]

    box = boxes[best_idx]
    score = scores[best_idx]
    mask = masks[best_idx, 0] > 0.5

    result = image_bgr.copy()

    # Create green mask overlay
    colored_mask = np.zeros_like(result)
    colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)
    result = cv2.addWeighted(result, 1.0, colored_mask, 0.4, 0)

    # Draw bounding box
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Label
    text = f"book: {score:.2f}"
    cv2.putText(
        result,
        text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return result, True


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
# Run inference
# ==========================================================
image_files = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_files)} test images")

with torch.no_grad():
    for image_name in image_files:
        image_path = os.path.join(TEST_DIR, image_name)

        pil_image = Image.open(image_path).convert("RGB")
        image_np = np.array(pil_image)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(DEVICE)

        outputs = model([image_tensor])[0]

        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        result, found = draw_best_prediction(image_bgr, outputs, SCORE_THRESHOLD)

        if not found:
            result = image_bgr.copy()
            cv2.putText(
                result,
                "No confident detection",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        output_path = os.path.join(OUTPUT_DIR, image_name)
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")

print("\nInference completed.")
print(f"Results saved in: {OUTPUT_DIR}")