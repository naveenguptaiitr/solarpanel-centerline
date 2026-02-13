import os
import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation

from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA

import io
import cairosvg

import cv2
import xml.etree.ElementTree as ET

def convert_to_binary_mask_2(label_path, stroke_thickness=3):
    
    tree = ET.parse(label_path)
    root = tree.getroot()

    # Create empty mask (same size as SVG canvas)
    H, W = 500, 500
    mask = np.zeros((H, W), dtype=np.uint8)

    # Parse each <path> element
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] 
        if tag == 'path':
            d = elem.attrib.get('d', '')
            stroke_width = int(float(elem.attrib.get('stroke-width', 1)))

            if d.startswith('M') and 'L' in d:
                parts = d.replace('M', '').replace('L', '').split()
                if len(parts) == 4:
                    x1, y1, x2, y2 = map(float, parts[:4])

                    pt1 = (int(round(x1)), int(round(y1)))
                    pt2 = (int(round(x2)), int(round(y2)))

                    cv2.line(mask, pt1, pt2, color=255, thickness=stroke_width+stroke_thickness)
    
    mask = (mask > 0).astype(np.uint8)
    return mask

def convert_to_binary_mask(label_path):

    H, W = 500, 500
    png_bytes = cairosvg.svg2png(
        url=label_path,
        output_width=W,
        output_height=H
    )

    svg_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(svg_img)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    mask = ((R > 150) & (G < 100) & (B < 100)).astype(np.uint8)
    mask = binary_dilation(mask, iterations=2).astype(np.uint8)
    return mask

class DataSplitStrategy:

    def simple_split(self, image_paths, label_paths, train_frac=0.8, val_frac=0.1):
        total_length = len(image_paths)

        train_image_paths = image_paths[0:int(train_frac * total_length)]
        train_label_paths = label_paths[0:int(train_frac * total_length)]

        val_image_paths = image_paths[int(train_frac * total_length):int((train_frac + val_frac) * total_length)]
        val_label_paths = label_paths[int(train_frac * total_length):int((train_frac + val_frac) * total_length)]

        test_image_paths = image_paths[int((train_frac + val_frac) * total_length):]
        test_label_paths = label_paths[int((train_frac + val_frac) * total_length):]

        return (train_image_paths, train_label_paths), (val_image_paths, val_label_paths), (test_image_paths, test_label_paths)

    def random_split(self, image_paths, label_paths, train_frac=0.8, val_frac=0.1, seed=42):
        np.random.seed(seed)
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)

        train_indices = indices[0:int(train_frac * len(image_paths))]
        val_indices = indices[int(train_frac * len(image_paths)):int((train_frac + val_frac) * len(image_paths))]
        test_indices = indices[int((train_frac + val_frac) * len(image_paths)):]

        train_image_paths = [image_paths[i] for i in train_indices]
        train_label_paths = [label_paths[i] for i in train_indices]
        val_image_paths = [image_paths[i] for i in val_indices]
        val_label_paths = [label_paths[i] for i in val_indices]
        test_image_paths = [image_paths[i] for i in test_indices]
        test_label_paths = [label_paths[i] for i in test_indices]

        return (train_image_paths, train_label_paths), (val_image_paths, val_label_paths), (test_image_paths, test_label_paths)


class GeometricMetrics:

    def mask_to_lines(self, binary_mask, min_length=10):
        """
        Input:
            binary_mask: 2D numpy array (prediction or GT)
            min_length: ignore tiny fragments
        Returns:
            List of Numpy arrays, each (num_points, 2)
        """
        skeleton = skeletonize(binary_mask > 0)
        labeled = label(skeleton)

        lines = []
        for region in regionprops(labeled):
            coords = region.coords  # (row, col) format
            if len(coords) >= min_length:
                # Convert to (x, y) = (col, row)
                lines.append(np.array([[c[1], c[0]] for c in coords]))
        return lines

    def fit_line_pca(self,coords):
        pca = PCA(n_components=2)
        pca.fit(coords)
        direction = pca.components_[0]
        center = coords.mean(axis=0)
        return center, direction

    def angular_error(self, dir1, dir2):
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = dir2 / np.linalg.norm(dir2)
        cos_angle = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    def mean_perpendicular_distance(self,coords, center, direction):
        direction = direction / np.linalg.norm(direction)
        diffs = coords - center
        projections = diffs - np.outer(np.dot(diffs, direction), direction)
        distances = np.linalg.norm(projections, axis=1)
        return distances.mean()

    def match_lines(self,pred_lines, gt_lines, dist_thresh=20, angle_thresh=10):
        """
        Returns row-level precision and recall
        Thresholds are in pixels and degrees
        """
        matches = 0
        used_gt = set()

        for pred in pred_lines:
            c1, d1 = self.fit_line_pca(pred)

            for i, gt in enumerate(gt_lines):
                if i in used_gt:
                    continue

                c2, d2 = self.fit_line_pca(gt)
                ang_err = self.angular_error(d1, d2)
                dist_err = self.mean_perpendicular_distance(pred, c2, d2)

                if ang_err < angle_thresh and dist_err < dist_thresh:
                    matches += 1
                    used_gt.add(i)
                    break
                
        precision = matches / len(pred_lines) if pred_lines else 0
        recall = matches / len(gt_lines) if gt_lines else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision+recall)>0 else 0

        return precision, recall, f1

    def evaluate_centerlines_pixel(self, pred_mask, gt_mask, dist_thresh=20, angle_thresh=10):
        """
        Input:
            pred_mask: 2D numpy array (predicted centerline mask)
            gt_mask: 2D numpy array (ground truth centerline mask)
            dist_thresh: pixel threshold for row-level matching
            angle_thresh: angular threshold (deg) for row-level matching
        Returns:
            Dictionary with pixel-based metrics
        """
        pred_lines = self.mask_to_lines(pred_mask)
        gt_lines = self.mask_to_lines(gt_mask)

        angular_errors = []
        perp_distances = []

        # Compute per-line metrics for matched lines
        for pred in pred_lines:
            c1, d1 = self.fit_line_pca(pred)
            min_ang = float('inf')
            min_dist = float('inf')

            for gt in gt_lines:
                c2, d2 = self.fit_line_pca(gt)
                ang_err = self.angular_error(d1, d2)
                dist_err = self.mean_perpendicular_distance(pred, c2, d2)

                if ang_err < min_ang:
                    min_ang = ang_err
                if dist_err < min_dist:
                    min_dist = dist_err

            angular_errors.append(min_ang)
            perp_distances.append(min_dist)

        # Row-level metrics
        precision, recall, f1 = self.match_lines(pred_lines, gt_lines, dist_thresh, angle_thresh)

        results = {
            "Mean Perpendicular Distance": np.mean(perp_distances) if perp_distances and not any(x == float('inf') for x in perp_distances) else 0,
            "Mean Angular Error": np.mean(angular_errors) if angular_errors and not any(x == float('inf') for x in angular_errors) else 0,
            "Row Precision": precision,
            "Row Recall": recall,
            "Row F1": f1
        }

        return results


class CVMetrics:

    def compute_iou(self, preds, targets, eps=1e-6):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection

        return (intersection + eps) / (union + eps)


    def compute_dice(self, preds, targets, eps=1e-6):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        return (2 * intersection + eps) / (preds.sum() + targets.sum() + eps)


    def compute_precision_recall_f1(self, preds, targets, eps=1e-6):
        preds = preds.view(-1)
        targets = targets.view(-1)

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return precision, recall, f1

def plot_preds_vs_gt(pred_masks, gt_masks, gt_images, num_images=5):
    """
    preds: Tensor or numpy array of shape (num_images, H, W)
    gts:   Tensor or numpy array of same shape
    """

    fig, axes = plt.subplots(num_images, 3, figsize=(8, 15))
    gt_images = np.transpose(gt_images, (0, 2, 3, 1))  # Convert to (N, H, W, C)

    for i in range(num_images):
        # Prediction
        axes[i, 0].imshow(pred_masks[i].numpy())
        axes[i, 0].imshow(gt_images[i].numpy(), alpha=0.6, cmap="Blues")
        axes[i, 0].set_title(f"Prediction_{i}")
        axes[i, 0].axis("off")

        # Ground truth
        axes[i, 1].imshow(gt_masks[i].numpy())
        axes[i, 1].imshow(gt_images[i].numpy(), alpha=0.6, cmap="Reds")
        axes[i, 1].set_title(f"Ground Truth_{i}")
        axes[i, 1].axis("off")

        # Error
        axes[i, 2].imshow(pred_masks[i].numpy(), cmap='Blues', alpha=0.6)
        axes[i, 2].imshow(gt_masks[i].numpy(), cmap='Reds', alpha=0.6)
        axes[i, 2].set_title(f"Mask Comparison_{i}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()
