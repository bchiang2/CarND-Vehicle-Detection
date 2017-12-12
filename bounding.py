import numpy as np
from scipy.ndimage.measurements import label
import cv2
from classifier.train import extract_features
from classifier.train import SVC, SCALER
from helper import get_windows_with_scale
from functools import reduce
import config


class HeatmapQueue():
    def __init__(self):
        self.queue = []

    def append(self, new_heatmap):
        if len(self.queue) > config.HEATMAP_QUEUE_SIZE:
            self.queue.pop(0)
        self.queue.append(new_heatmap)

    def sum(self):
        return reduce(lambda x, y: np.add(x, y), self.queue)


heatmap_queue = HeatmapQueue()


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


i = 0


def search_windows(img, windows):
    on_windows = []
    for window in windows:
        global i
        i += 1
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = extract_features(test_img)
        test_features = SCALER.transform(np.array(features).reshape(1, -1))
        prediction = SVC.predict(test_features)
        if prediction == 1:
            # cv2.imwrite('box_{}.png'.format(i), cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
            on_windows.append(window)
    return on_windows


def get_heat_map(image, boxes):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, boxes)
    return apply_threshold(heat, 1)


def get_labeled_bbox(image):
    windows = get_windows_with_scale(image)
    heatmap = get_heat_map(image, search_windows(image, windows))
    heatmap_queue.append(heatmap)
    labels = label(heatmap_queue.sum())
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return image


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy



def get_boxes(image):
    windows = get_windows_with_scale(image)
    return draw_boxes(image, search_windows(image, windows))

def get_heat(image):
    windows = get_windows_with_scale(image)
    return get_heat_map(image, search_windows(image, windows))

def get_label(image):
    get_heat(image)