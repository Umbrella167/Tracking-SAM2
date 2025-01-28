import dearpygui.dearpygui as dpg

from ui.DynamicLoader import DynamicLoader
from ui.LayoutManager import LayoutManager
from utils.Utils import get_all_subclasses
from ui.boxes import *


class UI:
    def __init__(self):
        dpg.create_context()
        self.layout_manager = LayoutManager(ui=self)
        self.boxes = []
        self.box_count = {}
        self.is_created = False
        self.console_box = None
        self.input_box = None
        self.dl = DynamicLoader()
        self.all_classes = self.dl.boxes
        # self.generate_add_methods()
        self.layout_init_file = "static/layout/layout_init.json"

    def create(self):
        self.layout_manager.load()
        self.add_global_handler()
        self.create_viewport_menu()
        self.new_box("ConsoleBox")
        self.new_box("InputConsoleBox")
        self.is_created = True

    def create_viewport_menu(self):
        pass
        # with dpg.viewport_menu_bar():
        #     with dpg.menu(label="File", tag="file_menu"):
        #         dpg.add_menu_item(label="Save Layout")
        #         dpg.add_menu_item(label="Load Layout")
        #         dpg.add_menu_item(label="Exit")
        #     with dpg.menu(label="SSL3D", tag="ssl3d_menu"):
        #         dpg.add_menu_item(
        #             label="Camera option  (F2)",
        #             tag="camera_option_menutiem",
        #             # callback=self.pop_camera_option_window,
        #         )

    def show(self):
        if not self.is_created:
            self.create()

    def update(self):
        for box in self.boxes:
            if box.is_created:
                box.update()

    def new_box(self, box_name, **kwargs):
        cls = self.all_classes[box_name]
        instance = cls(ui=self, **kwargs)
        instance.create_box()

    def destroy_all_boxes(self):
        for box in self.boxes:
            box.destroy()

    def run(self):
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            try:
                self.update()
                self.dl.check_boxes()
            except Exception as e:
                client_logger.log("ERROR", f"Loop Failed!", e)
        else:
            self.destroy_all_boxes()

    def add_global_handler(self):
        # 创建全局监听
        with dpg.handler_registry():
            # 松开按键时监听
            dpg.add_key_release_handler(callback=self.on_key_release)
            # 鼠标移动检测
            dpg.add_mouse_move_handler(callback=self.on_mouse_move)
            # # 鼠标滚动检测
            # dpg.add_mouse_wheel_handler(callback=self.on_mouse_wheel)

    # # 生成添加类方法
    # def generate_add_methods(self):
    #     for cls in self.all_classes.values():
    #         method_name = f"add_{cls.__name__}"
    #         # 使用闭包捕获cls
    #         def add_method(self, cls=cls, **kwargs):
    #             try:
    #                 if cls.only and self.box_count.get(cls, 0) >= 1:
    #                     # 如果盒子已经创建则不重复创建
    #                     raise Exception("This box can only be created once")
    #                 instance = cls(**kwargs)
    #                 instance.create_box()
    #                 return instance
    #             except Exception as e:
    #                 client_logger.log("WARNING", f"Unable to instantiate {cls}", e=e)
    #         # 将生成的方法绑定到当前实例
    #         setattr(self, method_name, add_method.__get__(self))

    # 监听事件
    def on_key_release(self, sender, app_data, user_data):
        if dpg.is_key_down(dpg.mvKey_LControl) and app_data == dpg.mvKey_S:
            self.layout_manager.save()
            client_logger.log("SUCCESS", "Layout saved successfully!")
        if dpg.is_key_released(dpg.mvKey_F11):
            dpg.toggle_viewport_fullscreen()

    def on_mouse_move(self, sender, app_data):
        pass
        # self.ui_data.draw_mouse_pos_last = self.ui_data.draw_mouse_pos
        # self.ui_data.draw_mouse_pos = dpg.get_drawing_mouse_pos()
        # self.ui_data.mouse_move_pos = tuple(
        #     x - y for x, y in zip(self.ui_data.draw_mouse_pos, self.ui_data.draw_mouse_pos_last)
        # )
