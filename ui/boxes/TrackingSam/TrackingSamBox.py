from dearpygui import dearpygui as dpg
from ui.boxes.BaseBox import BaseBox
from ui.boxes.TrackingSam.core.tracking import TrackingSAM2
from ui.components.Canvas2D import Canvas2D
import cv2
import numpy as np
from ui.boxes.TrackingSam.tracker import TapirModelHandler
import time
from tapnet.utils import model_utils
from utils.ClientLogManager import client_logger
from ui.boxes.TrackingSam.poisson import draw_uniform_points
import threading
from ui.boxes.TrackingSam.tracker import NUM_POINTS
SIZE = (320, 240)
MODEL_PATH = "ui/boxes/TrackingSam/static/sam2.1_hiera_tiny.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
TAPIR_MODEL_PATH = "ui/boxes/TrackingSam/static/causal_tapir_checkpoint.npy"


class TrackingSamBox(BaseBox):
    only = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sam2 = TrackingSAM2(model_path=MODEL_PATH, model_config=MODEL_CONFIG)
        self.frame = None
        self.ret = None
        self.cap = None
        self.obj_marker = None
        self.polygon_tag = None
        self.tracking_obj_image = None
        self.is_tracking_start = False
        # Initialize Tapir model
        self.pos = [None] * NUM_POINTS  # Store positions for each point
        self.query_frame = False
        self.last_click_time = 0
        self.have_point = [False] * NUM_POINTS
        self.next_query_idx = 0
        self.tapir_model_handler = TapirModelHandler(TAPIR_MODEL_PATH)
        self.track_layer = None
        self.frame_layer = None

    def create(self):
        self.canvas = Canvas2D(parent=self.tag, auto_mouse_transfrom=False)
        self.frame_layer = self.canvas.add_layer()
        self.track_layer = self.canvas.add_layer()

        self.texture_tag = self.canvas.texture_register(SIZE, dpg.mvFormat_Float_rgb)
        # self.cap = cv2.VideoCapture("ui/boxes/TrackingSam/static/1.mp4")
        self.cap = cv2.VideoCapture(0)
        self.frame_thread()
        client_logger.log("INFO", "get first image...")
        while self.frame is None:
            ret, self.frame = self.cap.read()
            if not ret or self.frame is None:
                continue
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.frame = cv2.resize(self.frame, SIZE)
        client_logger.log("INFO", "tapir model init...")
        self.tapir_model_handler.initialize(self.frame)
        client_logger.log("INFO", "tapir model init done.")
        with self.canvas.draw(parent=self.frame_layer):
            draw_img_tag = dpg.draw_image(self.texture_tag, [0, 0], SIZE)
        self.handler_register()

    def handler_register(self):
        with dpg.handler_registry():
            dpg.add_mouse_double_click_handler(
                button=dpg.mvMouseButton_Left, callback=self.tracking, user_data=None
            )

    def tracking(self,sender,app_data,user_data):
        self.is_tracking_start = True
        self.mark_obj(user_data)

    def mark_obj(self,track_center = None,sam2_points = None):
        if track_center is None:
            real_mouse_pos = self.canvas.pos_apply_transform(dpg.get_drawing_mouse_pos())
        else:
            real_mouse_pos = self.canvas.pos_apply_transform(track_center)

        sam2_points = list(real_mouse_pos) if sam2_points is None else sam2_points
        x, y = real_mouse_pos

        # 清理旧标记
        self._cleanup_old_markers()

        # 生成新掩膜和采样点
        self.sam2.load_image(self.frame)
        mask, _, _ = self.sam2.add_positive_point([sam2_points])
        self.tracking_obj_image = cv2.bitwise_and(
            self.frame, self.frame, mask=mask.astype("uint8")
        )

        # 优化后的泊松采样
        img, points = draw_uniform_points(self.frame, mask, NUM_POINTS)

        # 批量跟踪处理
        self._batch_track_points(points)

        # 可视化优化
        self._visualize_initial_points(points)

    def _cleanup_old_markers(self):
        """统一清理旧标记"""
        for item in [self.obj_marker, self.polygon_tag]:
            if dpg.does_item_exist(item):
                dpg.delete_item(item)
        # Also clear the tracked points
        dpg.delete_item(self.track_layer, children_only=True)
        self.have_point = [False] * NUM_POINTS
        self.pos = [None] * NUM_POINTS

    def _batch_track_points(self, points):
        """批量处理点跟踪的核心逻辑"""
        if self.frame is None or not points:
            return

        # 获取可用索引
        available_indices = [i for i, has in enumerate(self.have_point) if not has]

        # 确定实际可跟踪的点数
        valid_count = min(len(points), len(available_indices))
        if valid_count == 0:
            client_logger.log("WARNING", "No available tracking slots")
            return

        # 准备批量数据
        selected_points = points[:valid_count]
        selected_indices = available_indices[:valid_count]

        # 转换坐标格式（注意OpenCV和DearPyGui坐标差异）
        # converted_points = [(p[1], self.frame.shape[1]-p[0]) for p in selected_points]
        converted_points = [(p[1], p[0]) for p in selected_points]

        try:
            # 使用优化后的批量跟踪方法
            self.tapir_model_handler.track_points(
                frame=self.frame,
                positions=converted_points,
                indices_to_update=selected_indices,
            )

            # 更新状态
            for idx in selected_indices:
                self.have_point[idx] = True
                self.pos[idx] = converted_points[selected_indices.index(idx)]

            client_logger.log("INFO", f"Successfully tracked {valid_count} points")

        except Exception as e:
            client_logger.log("ERROR", f"Tracking failed: {str(e)}")

    def _visualize_initial_points(self, points):
        """优化初始点可视化"""
        # 绘制初始采样点
        for point in points[:NUM_POINTS]:  # 确保不超过最大点数
            dpg.draw_circle(
                center=tuple(point),
                radius=5,
                color=(0, 0, 255, 255),
                fill=(0, 0, 255, 255),
                parent=self.track_layer,
            )
        # 标记点击位置
        self.obj_marker = dpg.draw_circle(
            center=dpg.get_drawing_mouse_pos(),
            radius=7,
            color=(255, 0, 0, 255),
            fill=(255, 0, 0, 255),
            thickness=2,
            parent=self.track_layer,
        )

    def visualize_tracks(self, prediction):
        """优化跟踪可视化逻辑"""
        if prediction is None:
            return
        dpg.delete_item(self.track_layer, children_only=True)
        # 解析预测结果
        track = np.round(prediction["tracks"][0, :, 0])
        occlusion = prediction["occlusion"][0, :, 0]
        expected_dist = prediction["expected_dist"][0, :, 0]
        visibles = model_utils.postprocess_occlusions(occlusion, expected_dist)

        # 批量绘制可见点
        tracked_points = 0
        with self.canvas.draw():
            for i in range(NUM_POINTS):
                if visibles[i] and self.have_point[i]:
                    # 坐标转换（YOLO到DearPyGui格式）
                    vis_x = track[i, 0]
                    vis_y = track[i, 1]

                    # 动态颜色根据跟踪置信度
                    confidence = 1 - expected_dist[i]
                    color = (
                        int(255 * (1 - confidence)),  # R
                        int(255 * confidence),  # G
                        0,  # B
                        255,  # A
                    )

                    dpg.draw_circle(
                        center=(vis_x, vis_y),
                        radius=5,
                        color=color,
                        fill=color,
                        parent=self.track_layer,
                    )
                    tracked_points += 1
        predicted_rate = tracked_points / NUM_POINTS
        print(f"predicted rate {predicted_rate} %")
        # if predicted_rate < 0.7:
        #     pass
        if predicted_rate < 0.7 and self.is_tracking_start:
            # Calculate the center of all tracked points
            valid_points = [self.pos[i] for i in range(NUM_POINTS) if self.have_point[i]]
            # self.mark_obj(None, valid_points)
            valid_points = [ [x.item(), y.item()] for x, y in valid_points ]
            print(valid_points)
            
            # if valid_points:
            #     center_x = sum(p[0] for p in valid_points) / len(valid_points)
            #     center_y = sum(p[1] for p in valid_points) / len(valid_points)
            #     center = (center_x, center_y)
            #     # Re-track from the center
            #     self.mark_obj(center)
            #     client_logger.log("INFO", "Re-tracking from the center of tracked points.")
            # else:
            #     client_logger.log("WARNING", "No valid tracked points to re-track from.")

    def frame_get(self):
        while True:
            self.ret, frame = self.cap.read()
            if not self.ret:
                continue
            if frame is None:
                continue

            self.frame = cv2.cvtColor(cv2.resize(frame, SIZE), cv2.COLOR_BGR2RGB)

    def frame_thread(self):
        thread = threading.Thread(target=self.frame_get)
        thread.start()

    def update(self):
        """优化帧更新逻辑"""
        # ret, frame = self.cap.read()

        if self.ret is None or self.frame is None:
            return
        # 预处理帧
        # self.frame = cv2.cvtColor(cv2.resize(frame, SIZE), cv2.COLOR_BGR2RGB)

        try:
            self.canvas.texture_update(self.texture_tag, self.frame)
            # 执行预测
            prediction = self.tapir_model_handler.predict(self.frame)
            # 更新可视化
            self.visualize_tracks(prediction)
            # 更新纹理

        except Exception as e:
            client_logger.log("ERROR", f"Frame update failed: {str(e)}")