import dearpygui.dearpygui as dpg
from ui.boxes import BaseBox

class InputConsoleBox(BaseBox):
    only = True
    save = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.width = 1000
        self.height = 300
        self.is_sticky = False
        self.all_classes = self.ui.all_classes.values()
        self.old_classes = list(self.all_classes)
        self.input_text = None
        self.select_index = 0
        # 筛选后的列表
        self.filter_list = None
        self.selectables = {}

    def create(self):
        # 初始化设置
        dpg.configure_item(
            self.tag,
            width=self.width,
            height=self.height,
            # autosize=True,
            # collapsed=False,
            # no_resize=True,
            no_move=True,
            no_collapse=True,
            no_close=True,
            no_saved_settings=True,
            no_title_bar=True,
            popup=not self.is_sticky,
            show=False,
        )
        self.input_text = dpg.add_input_text(
            width=self.width, height=self.height, callback=self.filter_res, parent=self.tag
        )
        self.generate_selectables()

    def generate_selectables(self):
        for cls in self.all_classes:
            if not cls.save:
                continue
            self.selectables[cls.__name__] = dpg.add_selectable(
                label=cls.__name__,
                filter_key=cls.__name__,
                callback=lambda s,a,u:self.ui.new_box(u),
                parent=self.tag,
                user_data=cls.__name__,
            )

    def clear_all_selectables(self):
        list(map(dpg.delete_item, self.selectables.values()))
        self.selectables.clear()
        self.generate_selectables()
        self.select_index = 0
        self.filter_res(None, "")

    def filter_res(self, sender, app_data):
        self.select_index = 0
        [dpg.hide_item(i) for i in self.selectables.values()]
        self.filter_list = []
        for cls in self.all_classes:
            if not cls.save:
                continue
            pos = 0
            match = True
            for char in app_data.lower():
                pos = cls.__name__.lower().find(char, pos)
                if pos == -1:
                    match = False
                    break
                pos += 1
            if match:
                dpg.show_item(self.selectables[cls.__name__])
                self.filter_list.append(cls.__name__)
        self.update_selected()

    def key_release_handler(self, sender, app_data, user_data):
        key = app_data
        if not dpg.is_item_visible(self.tag) and dpg.is_key_down(dpg.mvKey_LControl) and dpg.is_key_released(dpg.mvKey_T):
            dpg.set_item_pos(
                self.tag,
                [dpg.get_viewport_width() / 2 - self.width / 2, dpg.get_viewport_height() / 2 - self.height / 2],
            )
            dpg.set_value(self.input_text, "")
            self.filter_res(None, "")
            self.show()
        if key == dpg.mvKey_Escape and not self.is_sticky:
            self.hide()
        if self.filter_list:
            if dpg.is_item_visible(self.tag) and key == dpg.mvKey_Return:
                self.ui.new_box(self.filter_list[self.select_index])
        self.update_selected()
        dpg.focus_item(self.input_text)

    def key_press_handler(self, sender, app_data, user_data):
        key = app_data
        if dpg.is_item_visible(self.tag):
            if key == dpg.mvKey_Up:
                self.select_index -= 1
                if self.select_index < 0:
                    self.select_index = len(self.filter_list) - 1
            if key == dpg.mvKey_Down:
                self.select_index += 1
                if self.select_index == len(self.filter_list):
                    self.select_index = 0
            self.update_selected()

    def update_selected(self):
        if not self.filter_list:
            return
        for selectable in self.selectables.values():
            dpg.configure_item(selectable, default_value=False)
        cls_name = self.filter_list[self.select_index]
        dpg.set_value(self.selectables[cls_name], True)
        dpg.configure_item(self.input_text, hint=cls_name)

    def update(self):
        if self.old_classes != list(self.all_classes):
            self.clear_all_selectables()
            self.old_classes = list(self.all_classes)

    def destroy(self):
        super().destroy()
