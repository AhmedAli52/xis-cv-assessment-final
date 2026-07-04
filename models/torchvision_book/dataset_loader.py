import os
import json
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


class BookCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.transforms = transforms

        self.coco = COCO(annotation_file)

        # Keep only images that actually have annotations
        self.image_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            if len(anns) > 0:
                self.image_ids.append(img_id)

        # Map all existing category IDs to ONE class: book -> label 1
        self.category_id_map = {}
        for cat_id in self.coco.cats.keys():
            self.category_id_map[cat_id] = 1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # Skip invalid boxes
            if w <= 1 or h <= 1:
                continue

            boxes.append([x, y, x + w, y + h])

            # Map any class from Roboflow to a single "book" class
            labels.append(1)

            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

            segmentation = ann.get("segmentation", [])

            if isinstance(segmentation, list):
                rles = coco_mask.frPyObjects(segmentation, height, width)
                rle = coco_mask.merge(rles)
                mask = coco_mask.decode(rle)
            else:
                mask = self.coco.annToMask(ann)

            masks.append(mask)

        if len(boxes) == 0:
            # fallback empty target
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))