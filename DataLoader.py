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
            #test thing
            area1 = abs(x2 - x1) * abs(y2 - y1)
            area2 = abs(adj_x2-adj_x1) * (adj_y2-adj_y1)
            if area1*.15 > area2:
                continue
            #print(adj_x1,adj_y1,adj_x1,adj_y2,x1,x2,y1,y2)
            
            
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


import cv2
import numpy as np
import matplotlib.pyplot as plt

import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def combine_tiles_grid_auto(tile_tensors, targets, fill_color=(0,0,0)):
    """
    Visualize multiple tiles with bounding boxes in an automatic grid.

    tile_tensors: list of tensors [C,H,W]
    targets: list of dicts with 'boxes' and 'labels'
    fill_color: color to fill empty spots if tiles < grid_size
    """
    n_tiles = len(tile_tensors)
    if n_tiles == 0:
        raise ValueError("No tiles provided")

    # Compute grid size (rows x cols) to form closest-to-square grid
    cols = math.ceil(math.sqrt(n_tiles))
    rows = math.ceil(n_tiles / cols)

    processed_imgs = []

    # Process each tile
    for idx, (tile_tensor, target) in enumerate(zip(tile_tensors, targets)):
        img = tile_tensor.permute(1,2,0).numpy()
        img = (img * 255).astype(np.uint8).copy()

        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()

        for i, (x_center, y_center, bw, bh) in enumerate(boxes):
            x1 = int((x_center - bw / 2) * img.shape[1])
            y1 = int((y_center - bh / 2) * img.shape[0])
            x2 = int((x_center + bw / 2) * img.shape[1])
            y2 = int((y_center + bh / 2) * img.shape[0])
            color = (0, 255, 0)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, str(labels[i]), (x1 + 3, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        processed_imgs.append(img)

    # If number of tiles < grid_size, pad with empty images
    tile_h, tile_w = processed_imgs[0].shape[:2]
    empty_tile = np.full((tile_h, tile_w, 3), fill_color, dtype=np.uint8)
    while len(processed_imgs) < rows * cols:
        processed_imgs.append(empty_tile.copy())

    # Stack into grid
    grid_rows = []
    for r in range(rows):
        row_imgs = processed_imgs[r*cols:(r+1)*cols]
        # Ensure same height
        min_h = min(img.shape[0] for img in row_imgs)
        row_imgs = [cv2.resize(im, (im.shape[1]*min_h//im.shape[0], min_h)) for im in row_imgs]
        grid_rows.append(np.vstack(row_imgs))

    grid_img = np.hstack(grid_rows)

    # Display
    plt.figure(figsize=(cols*3, rows*3))
    plt.imshow(grid_img)
    plt.axis('off')
    plt.show()

def patch_tiles_from_dataset(dataset, img_id):
    """
    Reconstructs (stitches) all tiles belonging to a specific image_id 
    into a full-size composite image with their annotations.

    Returns:
        full_img (np.ndarray): reconstructed RGB image
        full_targets (list): combined targets (boxes, labels) in original coordinates
    """
    # Get all tiles for this image_id
    relevant_tiles = [(idx, x, y) for idx, (imgid, x, y, _, _) in enumerate(dataset.tiles) if imgid == img_id]
    if not relevant_tiles:
        raise ValueError(f"No tiles found for image_id {img_id}")

    # Sort tiles by position (top-to-bottom, left-to-right)
    relevant_tiles.sort(key=lambda t: (t[2], t[1]))  # sort by y, then x

    # Get original image info
    img_info = dataset.images[img_id]
    w, h = img_info["width"], img_info["height"]

    tile_size = dataset.tile_size
    stride = dataset.stride

    # Determine how many tiles fit horizontally and vertically
    n_cols = math.ceil(w / stride)
    n_rows = math.ceil(h / stride)

    # Create empty canvas (reconstructed image)
    full_h = n_rows * tile_size
    full_w = n_cols * tile_size
    full_img = np.zeros((full_h, full_w, 3), dtype=np.uint8)

    full_boxes = []
    full_labels = []
    
   
    
    for idx, (tile_idx, x0, y0) in enumerate(relevant_tiles):
        tile_tensor, target = dataset[tile_idx]
        tile_img = tile_tensor.permute(1, 2, 0).numpy()
        tile_img = (tile_img * 255).astype(np.uint8).copy()

        # Paste tile in correct position
        full_img[y0:y0+tile_size, x0:x0+tile_size] = tile_img

        # Convert each box from tile-relative normalized coords to global
        for (xc, yc, bw, bh), label in zip(target['boxes'].numpy(), target['labels'].numpy()):
            abs_xc = x0 + xc * tile_size
            abs_yc = y0 + yc * tile_size
            abs_bw = bw * tile_size
            abs_bh = bh * tile_size

            x1 = abs_xc - abs_bw / 2
            y1 = abs_yc - abs_bh / 2
            x2 = abs_xc + abs_bw / 2
            y2 = abs_yc + abs_bh / 2

            full_boxes.append([x1, y1, x2, y2])
            full_labels.append(label)

    # Crop to original image size
    full_img = full_img[:h, :w]

    full_targets = {
        "boxes": np.array(full_boxes, dtype=np.float32),
        "labels": np.array(full_labels, dtype=np.int64)
    }
    orig_anns = dataset.annotations.get(img_id, [])
    for ann in orig_anns:
        bx, by, bw, bh = ann['bbox']
        x1, y1 = bx, by
        x2, y2 = bx + bw, by + bh
        cv2.rectangle(full_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return full_img, full_targets

def visualize_patched_image(full_img, full_targets):
    """Show stitched image with all boxes."""
    img = full_img.copy()

    for (x1, y1, x2, y2), label in zip(full_targets['boxes'], full_targets['labels']):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = (0, 255, 0)
        color_red = (255,0,0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(label), (x1 + 3, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    tile_size = 640
    thickness = 1
    h,w = img.shape[:2] 
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            cv2.rectangle(img, (x, y), (min(x + tile_size, w), min(y + tile_size, h)), color_red, thickness)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

imgpath = 'C:\\Users\\owner\\Downloads\\Images'
jsonpath = 'C:\\Users\\owner\\Downloads\\Annotations\\test.json'

dataset = CocoTiledDataset(coco_json=jsonpath,img_dir=imgpath,tile_size=640,stride=640)


def border_red_green_visualization(image,target):

    return
# for _ in range(10):
# img, target = dataset.__getitem__(6)
# print(img.shape,target)

# for _ in range(10):
#     img, target = dataset.next_item()
#     print(img.shape,target)

# # Test: visualize a few random tiles
# # print(f"Total tiles: {len(dataset)}")# 
indix = 0
for indix in range(5):
    random_img_id = list(dataset.images.keys())[indix]q
    full_img, full_targets = patch_tiles_from_dataset(dataset, random_img_id)
    visualize_patched_image(full_img, full_targets)
# visualize_tiling_on_image(dataset, random_img_id)

# for idx in range(1,8):
#     tile, target = dataset[idx]
#     print(f"Tile {idx}: {len(target['boxes'])} boxes")
#     visualize_tile(tile, target, idx)

# tile_tensors = []
# targets = []
# h,w = random_img_id.shape[:2]
# # Pick first N tiles
# N = 17


# for idx in range(20,32):
#     tile, target = dataset[idx]
#     tile_tensors.append(tile)
#     targets.append(target)

# combine_tiles_grid_auto(tile_tensors, targets)
# Pick a random image id from the dataset