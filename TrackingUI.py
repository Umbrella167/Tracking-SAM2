import dearpygui.dearpygui as dpg
import numpy as np
import Utils as utils
from SAM2 import sam2
import time
import cv2
class Share:
    def __init__(self):
        self.points = []
        self.labels = []
        self.points = []
        self.labels = []
        self.IMAGE_SIZE = (1080, 720)
        self.image = np.zeros((self.IMAGE_SIZE[1], self.IMAGE_SIZE[0], 3), dtype=np.uint8)
        self.texture = utils.image2texture(self.image)
    def load_sam2_image(self,image):
        sam2.load_image(image)
class UICallBack:
    def __init__(self,Share):
        self._share = Share
    def on_mouse_double_click(self, sender, app_data):
        sam2.load_image(self._share.image)
        mouse_x, mouse_y = dpg.get_drawing_mouse_pos()
        dpg.draw_circle(parent="pinter",center=[mouse_x, mouse_y], radius=5,color=[255, 0, 0, 255],fill=[255, 0, 0, 255])
        points = [[mouse_x, mouse_y]]
        labels = [1]
        masks, scores, logits = sam2.add_point(points,labels)
        sam2.mask2rect(masks, self._share.image)
        print(masks)
        cv2.imshow("mask", masks)
        cv2.waitKey(1)
    def clear_canvas(self):
        dpg.delete_item("pinter",children_only=True)
class UI:
    def __init__(self):
        self._share = Share()
        self._callback = UICallBack(self._share)
        dpg.create_context()
    def texture_registry(self, show=False):
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
            width=self._share.IMAGE_SIZE[0],
            height=self._share.IMAGE_SIZE[1],
            default_value=self._share.texture,
            tag="image_texture",
            format=dpg.mvFormat_Float_rgb,
            )
    def update_window(self):
        width, height = dpg.get_item_rect_size("main_window")
        height = height - 20
        dpg.configure_item(item="main_drawlist", width=width, height=height)
        dpg.draw_image(
            parent="canvas", texture_tag="image_texture", pmin=[0, 0], pmax=[self._share.IMAGE_SIZE[0], self._share.IMAGE_SIZE[1]]
        )
    def create_window(self):
        with dpg.window(label="Main",tag="main_window"):
            with dpg.drawlist(width=-1, height=-1, tag="main_drawlist"):
                with dpg.draw_node(tag="canvas"):
                    dpg.draw_circle([0, 0], 50)
                with dpg.draw_node(tag="pinter"):
                    dpg.draw_circle([0, 0], 50)
    def handler_registry(self):
        with dpg.handler_registry() as global_hander:
            dpg.add_mouse_double_click_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._callback.on_mouse_double_click,
            )
            dpg.add_key_release_handler(
                key=dpg.mvKey_Escape,
                callback=self._callback.clear_canvas,
            )
    def update_image(self,image):
        self._share.image = image
        self.texture = utils.image2texture(self._share.image)
        dpg.set_value("image_texture", self.texture)
    def start(self,loop):
        dpg.create_viewport(title="TEST", width=600, height=400)
        dpg.setup_dearpygui()
        self.handler_registry()
        self.texture_registry()
        self.create_window()
        
        dpg.set_primary_window("main_window", True)
        dpg.show_viewport()
        while dpg.is_dearpygui_running:
            FPS = dpg.get_frame_rate()
            dpg.set_viewport_title(f"Tracking-Test     FPS: {FPS}")
            self.update_window()
            loop()
            dpg.render_dearpygui_frame()

if __name__ == "__main__":
    ui = UI()
    ui.start()
