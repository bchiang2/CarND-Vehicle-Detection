import cv2
import numpy as np
from helper import get_windows_with_scale, over_lay_image_to_rgb
from models.video import Video
from bounding import get_labeled_bbox, get_boxes, get_heat

i = 0


def process_image(image):
    global i
    i += 1
    img = get_labeled_bbox(image)
    cv2.imwrite("final_{}.jpg".format(i), img)
    # image = over_lay_image_to_rgb(
    #     rgb=image,
    #     image=get_labeled_bbox(image),
    #     alpha=0.8,
    #     beta=1,
    #     weights=(255, 0, 0))
    # cv2.imwrite("box_{}.jpg".format(i), image)
    return img


def main():
    video = Video('test_video.mp4')
    video.play_video(image_function=process_image)


main()
