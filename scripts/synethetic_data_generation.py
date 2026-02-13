import cv2
import numpy as np
import random
import os

from PIL import Image

IMAGE_SIZE = 500
NUM_TRACKERS_MIN = 5
NUM_TRACKERS_MAX = 10
TRACKER_WIDTH = 8
TRACKER_LENGTH_MIN = 300
TRACKER_LENGTH_MAX = 450
OUTPUT_DIR = "synthetic_masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_procedural_mask():
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    num_trackers = random.randint(NUM_TRACKERS_MIN, NUM_TRACKERS_MAX)
    spacing = IMAGE_SIZE // (num_trackers + 1)
    
    for i in range(num_trackers):
        y_center = spacing * (i + 1) + random.randint(-5, 5)
        length = random.randint(TRACKER_LENGTH_MIN, TRACKER_LENGTH_MAX)
        angle = random.uniform(-10, 10)
        rect_center = (IMAGE_SIZE // 2, y_center)
        rect_size = (length, TRACKER_WIDTH)
        rotation_matrix = cv2.getRotationMatrix2D(rect_center, angle, 1.0)

        tracker_layer = np.zeros_like(mask)
        cv2.rectangle(tracker_layer,
                      (rect_center[0]-length//2, rect_center[1]-TRACKER_WIDTH//2),
                      (rect_center[0]+length//2, rect_center[1]+TRACKER_WIDTH//2),
                      255, -1)
        tracker_layer = cv2.warpAffine(tracker_layer, rotation_matrix, (IMAGE_SIZE, IMAGE_SIZE))
        mask = np.maximum(mask, tracker_layer)
    
    return mask
