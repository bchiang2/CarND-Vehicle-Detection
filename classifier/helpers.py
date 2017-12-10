import matplotlib.image as mpimg
from scipy.misc import imresize
import cv2
import numpy as np
import config


def convert_color(image, color_space):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            try:
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            except:
                print(image)
    else:
        feature_image = np.copy(image)
    return feature_image


def open_image_file(file_path, color_space=config.COLOR_SPACE):
    image = mpimg.imread(file_path)
    image = imresize(image, (64,64, 3))
    return convert_color(image, color_space)


def normalize_array(array):
    return array
    # max = np.max(array)
    # if max == 0:
    #     return array
    # else:
    #     return array * 1.0 / max
