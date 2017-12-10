import cv2
from classifier.train import extract_features
import numpy as np
from classifier.train import SVC, SCALER
from helper import get_windows_with_scale
from models.video import Video


def search_windows(img, windows):
    on_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = extract_features(test_img)
        test_features = SCALER.transform(np.array(features).reshape(1, -1))
        prediction = SVC.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def process_image(image):
    windows = get_windows_with_scale(image)
    boxes = search_windows(image, windows)
    return draw_boxes(image, boxes)


# for image_path in glob.glob('./test_images/*.jpg'):
#     print(image_path)
#     img = open_image_file(image_path)
#     windows = get_windows_with_scale(img)
#     boxes = search_windows(img, windows)
#     new_image = draw_boxes(img, boxes)
#     plt.imshow(new_image)
#     plt.show()

def main():
    video = Video('project_video.mp4')
    video.play_video(image_function=process_image)

main()