from dearpygui import dearpygui as dpg
from ui.boxes.BaseBox import BaseBox
from ui.boxes.TrackingSam.core.tracking import TrackingSAM2
from ui.components.Canvas2D import Canvas2D
import cv2
import numpy as np

PREDICT_FRAME = 10
SIZE = (320,240)
MODEL_PATH = "ui/boxes/TrackingSam/static/sam2.1_hiera_tiny.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

class TrackingSamBox(BaseBox):
    only = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sam2 = TrackingSAM2(model_path=MODEL_PATH, model_config=MODEL_CONFIG)
        self.frame = None
        self.cap = None
        self.obj_marker = None
        self.polygon_tag = None
        self.tracking_obj_image = None
    def create(self):        
        self.canvas = Canvas2D(parent=self.tag)
        self.texture_tag = self.canvas.texture_register(SIZE,dpg.mvFormat_Float_rgb)
        # self.cap = cv2.VideoCapture("ui/boxes/TrackingSam/static/1.mp4")
        self.cap = cv2.VideoCapture(0)
        
        with self.canvas.draw():
            dpg.draw_image(self.texture_tag, [0,0], SIZE)
        self.handler_register()

    def handler_register(self):
        with dpg.handler_registry():
            dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left,callback=self.tracking)
    def mask_to_polygon(self, mask):
        """
        将nparray格式的mask转换为多边形点的列表。
        
        参数:
            mask: np.ndarray, 二值化的mask数组 (可以是浮点型或其他非uint8类型)
        
        返回:
            polygons: 一个包含多边形点坐标列表的列表。
                    每个多边形是一个点坐标的列表,例如 [[(x1, y1), (x2, y2), ...], ...]
        """

        # 检查并处理 mask 数据类型,并确保其为二值化的 uint8 数据
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

        # 使用 OpenCV 进行二值化 (门限处理)
        binary_mask = cv2.inRange(mask, 127, 255)  # 更高效的二值化操作

        # 提取外部轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 提取多边形点坐标
        return [
            [(int(point[0][0]), int(point[0][1])) for point in contour]
            for contour in contours if len(contour) >= 3
        ]
    
    
    
    def mark_obj(self):
        real_mouse_pos = self.canvas.pos_apply_transform(dpg.get_drawing_mouse_pos())
        if dpg.does_item_exist(self.obj_marker):
            dpg.delete_item(self.obj_marker)
        if dpg.does_item_exist(self.polygon_tag):
            dpg.delete_item(self.polygon_tag)
        self.sam2.load_image(self.frame)
        mask, scores, logits = self.sam2.add_positive_point([list(real_mouse_pos)])
        self.tracking_obj_image = cv2.bitwise_and(self.frame, self.frame, mask=mask.astype('uint8'))
        
        
        with self.canvas.draw():
            self.obj_marker = dpg.draw_circle(center=real_mouse_pos, radius=5, color=(255,0,0,255), fill=(255,0,0,255))
            
        
    def tracking(self):
        self.mark_obj()
        
    def update(self):
        ret,frame = self.cap.read()
        if frame is None:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, SIZE)
        self.frame = frame
        self.canvas.texture_update(self.texture_tag,self.frame)
        print(1)
        if self.tracking_obj_image is not None:
            # 获取关键点和描述子
            keypoints_frame, des_frame = self.sam2.orb_get_points(self.frame)
            keypoints_obj, des_obj = self.sam2.orb_get_points(self.tracking_obj_image)
            
            # 创建 BFMatcher 对象
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # 进行特征匹配
            matches = bf.match(des_frame, des_obj)
            
            # 按匹配距离排序
            matches = sorted(matches, key=lambda x: x.distance)
            
            # 计算匹配的置信度
            if matches:
                distances = [match.distance for match in matches if match.distance < 50]
                print(distances)
                avg_distance = sum(distances) / len(distances)
                max_distance = 200  # 最大可能距离,根据实际情况调整
                confidence = 1 - (avg_distance / max_distance)
                confidence = max(0, min(confidence, 1))  # 将置信度限制在 [0, 1] 范围内
                print(f"Confidence: {confidence * 100:.2f}%")
                
                result_img = cv2.drawMatches(frame, keypoints_frame, self.tracking_obj_image, keypoints_obj, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imshow("Matching Results", result_img)
                cv2.waitKey(1)

            else:
                print("No matches found.")