from TrackingUI import UI
import cv2
import time
import dearpygui.dearpygui as dpg


def loop():
    COUNT = dpg.get_frame_count()
    ret, frame = capture.read()
    frame = cv2.resize(frame, ui._share.IMAGE_SIZE)
    if COUNT % 10 == 0:
        pass
        # print("load image")
        # ui._share.load_sam2_image(frame)
    else:
        time.sleep(0.005)
    ui.update_image(frame)

if __name__ == "__main__":
    VEDIO_DIR = "2.webm"
    capture = cv2.VideoCapture(VEDIO_DIR)
    ui = UI()
    ui.start(loop)
    

