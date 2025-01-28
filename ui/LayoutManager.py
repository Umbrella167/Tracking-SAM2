import base64
import pickle

import dearpygui.dearpygui as dpg
import json

from config.SystemConfig import LAYOUT_CONFIG_FILE, THEME_PATH
from config import DynamicConfig
from utils.ClientLogManager import client_logger
from pathlib import Path

class LayoutManager:
    APP_EXCLUSIONS = ["version", "major_version", "minor_version", "platform", "device_name"]
    VIEWPORT_EXCLUSIONS = ["client_width", "client_height", "clear_color"]

    def __init__(self, ui, config_file=LAYOUT_CONFIG_FILE):
        self.ui = ui
        self.config_file = config_file
        # 从配置文件中加载配置
        self.config = self.load_config(config_file)
        self.interface_config = None
        self.box_default_layout = None

    def load(self):
        self.load_ui()
        self.load_boxes()

    def load_ui(self):
        ui_config = self.config.get("ui", {})
        viewport_config = ui_config.get("viewport", {
            "title": "DPG-Client",
            "width": 1920,
            "height": 1080,
        })
        dpg.create_viewport(
            **viewport_config
        )
        app_config = ui_config.get("app", {
            "docking": True,
            "docking_space": True,
        })
        dpg.configure_app(
            **app_config
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

        self.interface_config = ui_config.get("interface", {
            "theme": "dark",
            "language": "zh",
        })
        self.set_theme(self.interface_config["theme"])


    @staticmethod
    def load_config(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError as e:
            client_logger.log("WARNING", f"Configuration file {config_file} not found", e)
            return {}
        except json.JSONDecodeError as e:
            client_logger.log("WARNING", f"The configuration file {config_file} is in the wrong format", e)
            return {}

    def set_theme(self, theme):
        theme_path = THEME_PATH+theme+".json"
        theme_config = json.loads(open(theme_path, "r", encoding="utf-8").read())
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # 设置颜色
                for color_name, color_value in theme_config["colors"].items():
                    color_enum = getattr(dpg, color_name, None)
                    if color_enum:
                        dpg.add_theme_color(color_enum, color_value)
                # 设置样式
                for style_name, style_value in theme_config["styles"].items():
                    style_enum = getattr(dpg, style_name, None)
                    if style_enum:
                        if isinstance(style_value, list) and len(style_value) == 2:
                            dpg.add_theme_style(style_enum, style_value[0], style_value[1])
                        else:
                            dpg.add_theme_style(style_enum, style_value)
        dpg.bind_theme(global_theme)
        # 设置字体
        self.set_font(theme_config["font"])

    def set_font(self, font_config):
        font_file = font_config.get("file", None)
        font_size = font_config.get("size", 15)
        if font_file is None:
            client_logger.log("WARNING", "No font file provided")
            return
        # 创建字体注册器
        with dpg.font_registry():
            with dpg.font(font_file, font_size, pixel_snapH=True) as custom_font:
                # 添加字体范围提示
                font_hint = font_config.get("range_hint", "mvFontRangeHint_Chinese_Full")
                font_hint_enum = getattr(dpg, font_hint, None)
                if font_hint_enum:
                    dpg.add_font_range_hint(font_hint_enum)
        dpg.bind_font(custom_font)

    def load_boxes(self):
        boxes_config = self.config.get("boxes", {})
        self.box_default_layout = boxes_config.get("default", {
            "box_width": DynamicConfig.BOX_WIDTH or 1280,
            "box_height": DynamicConfig.BOX_HEIGHT or 720,
            "box_default_pos": DynamicConfig.BOX_DEFAULT_POS or (300, 50),
            "box_pos_offset": DynamicConfig.BOX_POS_OFFSET or 20,
        })

        for k, v in self.box_default_layout.items():
            setattr(DynamicConfig, k.upper(), v)

        instances = boxes_config.get("instances", {})
        for ins_config in instances:
            try:
                self.ui.new_box(
                    ins_config["cls_name"],
                    width=ins_config["width"],
                    height=ins_config["height"],
                    pos=ins_config["pos"],
                    data=pickle.loads(base64.b64decode(ins_config["data"].encode())),
                )
            except Exception as e:
                client_logger.log("ERROR", f"Box {ins_config['cls_name']} failed", e)

    def save(self):
        layout_dir = Path(LAYOUT_CONFIG_FILE).parent
        if not layout_dir.exists():
            layout_dir.mkdir(parents=True)

        with open(LAYOUT_CONFIG_FILE, "w+") as f:
            app_config = {k: v for k, v in dpg.get_app_configuration().items() if k not in self.APP_EXCLUSIONS}
            viewport_config = {k: v for k, v in dpg.get_viewport_configuration(0).items() if k not in self.VIEWPORT_EXCLUSIONS}
            config = {
                "ui": {
                    "app": app_config,
                    "viewport": viewport_config,
                    "interface": self.interface_config,
                },
                "boxes": {
                    "default":self.box_default_layout,
                    "instances": self.get_boxes_config(),
                },
            }
            f.write(json.dumps(config))
            f.flush()

    def get_boxes_config(self):
        boxes_config = []
        for box in self.ui.boxes:
            if not box.save:
                continue
            boxes_config.append(
                {
                    "cls_name": box.__class__.__name__,
                    "width": dpg.get_item_width(box.tag),
                    "height": dpg.get_item_height(box.tag),
                    "pos": dpg.get_item_pos(box.tag),
                    "data": base64.b64encode(pickle.dumps(box.data)).decode(),
                }
            )
        return boxes_config

