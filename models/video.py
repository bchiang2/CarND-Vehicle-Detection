import cv2
from moviepy.editor import VideoFileClip
import config


class Video(object):
    def __init__(self, video_path):
        self.video_path = video_path

    def play_video(self, image_function):
        cap = cv2.VideoCapture(self.video_path)

        while (cap.isOpened()):
            config.GLOBAL_FRAME_COUNT += 1
            ret, frame = cap.read()
            if config.GLOBAL_FRAME_COUNT > config.STARTING_FRAME:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = image_function(rgb)
                cv2.imshow('frame', cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def write_video(self, image_function, input=r'project_video.mp4', output=r'project_video_out.mp4'):
        clip = VideoFileClip(input)
        new_clip = clip.fl_image(image_function)
        new_clip.write_videofile(output, audio=False)
