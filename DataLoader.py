import torch
from torch.utils.data import Dataset
import cv2
import json
import numpy as np
import os
from itertools import product
from queue import Queue
from threading import Thread

class CocoTiledDataset(Dataset):
    def __init__(self, 
                 coco_json, 
                 img_dir, 
                 tile_size=640, 
                 stride=640, 
                 transforms=None, 
                 max_queue_size=100):
        """
        COCO dataset loader that pre-processes images into 640x640 tiles
        and stores them in a background queue for consumption.
        """
        with open(os.path.expanduser(coco_json), 'r') as f:
            coco = json.load(f)

        self.img_dir = os.path.expanduser(img_dir)
        self.tile_size = tile_size
        self.stride = stride
        self.transforms = transforms

        # Thread-safe queue
        self.queue = Queue(maxsize=max_queue_size)

        # Build image/annotation mappings
        self.images = {img['id']: img for img in coco['images']}
        self.annotations = {}
        for ann in coco['annotations']:
            if ann.get("iscrowd", 0) == 0:
                self.annotations.setdefault(ann['image_id'], []).append(ann)

        # Precompute all tiles
        self.tiles = []
        for img_id, img_info in self.images.items():
            w, h = img_info['width'], img_info['height']
            pad_w = (tile_size - w % tile_size) if w % tile_size != 0 else 0
            pad_h = (tile_size - h % tile_size) if h % tile_size != 0 else 0
            new_w, new_h = w + pad_w, h + pad_h

            x_steps = range(0, new_w, stride)
            y_steps = range(0, new_h, stride)
            for (x, y) in product(x_steps, y_steps):
                self.tiles.append((img_id, x, y, new_w, new_h))

        # Start background thread to fill queue
        self.worker = Thread(target=self._enqueue_tiles, daemon=True)
        self.worker.start()

    def _enqueue_tiles(self):
        """Background thread: processes and enqueues tiles."""
        for idx in range(len(self.tiles)):
            item = self._process_tile(idx)
            self.queue.put(item)

    def _process_tile(self, idx):
        """Processes one tile + adjusted annotations."""
        img_id, x0, y0, padded_w, padded_h = self.tiles[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        pad_bottom = padded_h - h
        pad_right = padded_w - w

        img = cv2.copyMakeBorder(
            img, 0, pad_bottom, 0, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # Crop tile
        tile = img[y0:y0 + self.tile_size, x0:x0 + self.tile_size]

        th, tw = tile.shape[:2]
        if th < self.tile_size or tw < self.tile_size:
            tile = cv2.copyMakeBorder(
                tile, 0, self.tile_size - th, 0, self.tile_size - tw,
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        # Process annotations
        anns = self.annotations.get(img_id, [])
        boxes, labels = [], []

        for ann in anns:
            bx, by, bw, bh = ann['bbox']
            x1, y1 = bx, by
            x2, y2 = bx + bw, by + bh

            tx1, ty1 = x0, y0
            tx2, ty2 = x0 + self.tile_size, y0 + self.tile_size

            # Intersection
            ix1 = max(x1, tx1)
            iy1 = max(y1, ty1)
            ix2 = min(x2, tx2)
            iy2 = min(y2, ty2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            adj_x1 = ix1 - tx1
            adj_y1 = iy1 - ty1
            adj_x2 = ix2 - tx1
            adj_y2 = iy2 - ty1

            bw_new = adj_x2 - adj_x1
            bh_new = adj_y2 - adj_y1
            x_center = adj_x1 + bw_new / 2
            y_center = adj_y1 + bh_new / 2

            # Normalize
            x_center /= self.tile_size
            y_center /= self.tile_size
            bw_new /= self.tile_size
            bh_new /= self.tile_size

            boxes.append([x_center, y_center, bw_new, bh_new])
            labels.append(ann['category_id'])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.transforms:
            augmented = self.transforms(image=tile, bboxes=boxes, labels=labels)
            tile = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['labels']

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        tile = tile.astype(np.float32) / 255.0
        tile = torch.from_numpy(tile).permute(2, 0, 1)

        return tile, target

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        """For PyTorch Dataloader compatibility (processes directly)."""
        return self._process_tile(idx)

    def next_item(self, block=True, timeout=None):
        """Retrieve next processed tile + annotations from queue."""
        return self.queue.get(block=block, timeout=timeout)


import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_tile(tile_tensor, target, tile_index):
    """Visualize one tile with bounding boxes."""
    # Convert tensor [C,H,W] → [H,W,C]
    img = tile_tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8).copy()

    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()

    # Draw boxes (unnormalize)
    for i, (x_center, y_center, bw, bh) in enumerate(boxes):
        x1 = int((x_center - bw / 2) * img.shape[1])
        y1 = int((y_center - bh / 2) * img.shape[0])
        x2 = int((x_center + bw / 2) * img.shape[1])
        y2 = int((y_center + bh / 2) * img.shape[0])
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(labels[i]), (x1 + 3, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    print(boxes,labels)
    # Show tile with annotations
    plt.figure(figsize=(5, 5))
    plt.title(f"Tile #{tile_index} with {len(boxes)} boxes")
    plt.imshow(img)
    plt.axis("off")
    plt.show()



def visualize_tiling_on_image(dataset, img_id):
    """
    Shows the original image with tile boundaries overlaid
    and ground-truth boxes (before tiling).
    """
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    img_info = dataset.images[img_id]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    tile_size = dataset.tile_size
    stride = dataset.stride

    # Draw tile grid
    for y in range(0, h, stride):
        cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)
    for x in range(0, w, stride):
        cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1)

    # Draw original annotation boxes (green)
    anns = dataset.annotations.get(img_id, [])
    for ann in anns:
        bx, by, bw, bh = ann['bbox']
        x1, y1 = int(bx), int(by)
        x2, y2 = int(bx + bw), int(by + bh)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(ann['category_id']), (x1 + 5, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(8, 8))
    plt.title(f"Image {img_info['file_name']} with {len(anns)} boxes")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


imgpath = 'C:\\Users\\owner\\Downloads\\Images'
jsonpath = 'C:\\Users\\owner\\Downloads\\Annotations\\test.json'

dataset = CocoTiledDataset(coco_json=jsonpath,img_dir=imgpath,tile_size=640,stride=640)
img, target = dataset[0]
print(img,target)


# Test: visualize a few random tiles
print(f"Total tiles: {len(dataset)}")

for idx in range(1,8):
    tile, target = dataset[idx]
    print(f"Tile {idx}: {len(target['boxes'])} boxes")
    visualize_tile(tile, target, idx)

# Pick a random image id from the dataset
# random_img_id = list(dataset.images.keys())[0]

# visualize_tiling_on_image(dataset, random_img_id)
