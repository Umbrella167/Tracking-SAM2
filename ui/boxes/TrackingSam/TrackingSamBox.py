from dearpygui import dearpygui as dpg
from ui.boxes.BaseBox import BaseBox
from ui.boxes.TrackingSam.core.SAM2 import SAM2Image
from ui.components.Canvas2D import Canvas2D
import cv2

SIZE = (640,480)
MODEL_PATH = "ui/boxes/TrackingSam/static/sam2.1_hiera_tiny.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
class TrackingSamBox(BaseBox):
    only = False
  
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sam2 = SAM2Image(model_path=MODEL_PATH, model_config=MODEL_CONFIG)
        self.frame = None
        self.cap = None
    def create(self):
        self.canvas = Canvas2D(parent=self.tag)
        self.texture_tag = self.canvas.texture_register(SIZE,dpg.mvFormat_Float_rgb)
        self.cap = cv2.VideoCapture(0)
        with self.canvas.draw():
            dpg.draw_image(self.texture_tag, [0,0], SIZE)
            
    def handler_register(self):
        dpg.add_mouse_double_click_handler(callback=self.mouse_double_click_handler, parent=self.tracking)
        
    def tracking(self):
        pass
        
    def update(self):
        ret,self.frame = self.cap.read()
        if self.frame is None:
            return
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.canvas.texture_update(self.texture_tag,self.frame)
        