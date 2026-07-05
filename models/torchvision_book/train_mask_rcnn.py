import os
import torch
import torchvision
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset_loader import BookCocoDataset, collate_fn


# ==========================================================
# Paths
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "labeled_dataset", "train")
VALID_DIR = os.path.join(PROJECT_ROOT, "dataset", "labeled_dataset", "valid")

TRAIN_JSON = os.path.join(TRAIN_DIR, "_annotations.coco.json")
VALID_JSON = os.path.join(VALID_DIR, "_annotations.coco.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Config
# ==========================================================
NUM_CLASSES = 2   # background + book
BATCH_SIZE = 1    # MX450 has 2GB VRAM, keep it small
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
NUM_WORKERS = 0   # safer on Windows

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Model
# ==========================================================
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(
        weights="DEFAULT",
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
# Datasets
# ==========================================================
train_dataset = BookCocoDataset(TRAIN_DIR, TRAIN_JSON)
valid_dataset = BookCocoDataset(VALID_DIR, VALID_JSON)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)

print(f"Train images: {len(train_dataset)}")
print(f"Valid images: {len(valid_dataset)}")
print(f"Using device: {DEVICE}")


# ==========================================================
# Training setup
# ==========================================================
model = get_model(NUM_CLASSES)
model.to(DEVICE)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)


# ==========================================================
# Validation loss helper
# ==========================================================
@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.train()  # needed because detection models return losses in train mode
    total_loss = 0.0
    count = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()
        count += 1

    return total_loss / max(count, 1)


# ==========================================================
# Training loop
# ==========================================================
best_model_path = os.path.join(
    OUTPUT_DIR,
    "best_mask_rcnn_book.pth"
)
best_val_loss = float("inf")
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for step, (images, targets) in enumerate(train_loader):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Step [{step+1}/{len(train_loader)}] "
            f"Loss: {losses.item():.4f}"
        )

    avg_train_loss = epoch_loss / max(len(train_loader), 1)
    avg_val_loss = evaluate_loss(model, valid_loader, DEVICE)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print("=" * 60)
    print(f"Epoch {epoch+1} completed")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Valid Loss: {avg_val_loss:.4f}")
    print("=" * 60)

    lr_scheduler.step()

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = os.path.join(OUTPUT_DIR, "best_mask_rcnn_book.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved to: {best_model_path}")

print("\nTraining completed.")

# ==========================================================
# Save training history
# ==========================================================
history = {
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "train_loss": train_losses,
    "validation_loss": val_losses
}

history_path = os.path.join(OUTPUT_DIR, "history.json")

with open(history_path, "w") as f:
    json.dump(history, f, indent=4)

print(f"Training history saved to: {history_path}")

# ==========================================================
# Save training log
# ==========================================================
log_path = os.path.join(OUTPUT_DIR, "training_log.txt")

with open(log_path, "w") as f:
    f.write("MODEL TRAINING REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Epochs          : {NUM_EPOCHS}\n")
    f.write(f"Batch Size      : {BATCH_SIZE}\n")
    f.write(f"Learning Rate   : {LEARNING_RATE}\n")
    f.write("Optimizer       : Adam\n")
    f.write("Scheduler       : StepLR\n")
    f.write(f"Best Validation Loss : {best_val_loss:.4f}\n\n")

    f.write("Epoch-wise Losses\n")
    f.write("-" * 40 + "\n")

    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
        f.write(
            f"Epoch {i}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Validation Loss = {val_loss:.4f}\n"
        )

print(f"Training log saved to: {log_path}")

# ==========================================================
# Save loss curve
# ==========================================================
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(train_losses) + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    range(1, len(val_losses) + 1),
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Mask R-CNN Training Loss")

plt.grid(True)
plt.legend()

plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")

plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Best model saved at: {best_model_path}")

print(f"Loss curve saved to: {plot_path}")
