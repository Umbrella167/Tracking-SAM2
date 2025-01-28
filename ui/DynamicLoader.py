import os
import importlib
import inspect

from ui.boxes import BaseBox

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

class DynamicLoader:
    def __init__(self):
        self.boxes = {}
        self.box_files = None
        self.boxes_dir = os.path.join(os.path.dirname(__file__), 'boxes')

        self.load_boxes()

    def check_boxes(self):
        new_box_files = os.listdir(self.boxes_dir)
        if self.box_files == new_box_files:
            return
        # TODO: 暂时这样写，可以检查是否增加或者减少做更具体的操作
        self.reload_boxes()

    def load_boxes(self):
        self.box_files = os.listdir(self.boxes_dir)
        for filename in self.box_files:
            # 排除特殊文件夹
            if '__' not in filename:
                if os.path.isdir(os.path.join(self.boxes_dir, filename)):
                    # 如果是文件夹
                    module_name = f'ui.boxes.{filename}'
                else:
                    # 如果是单个文件
                    module_name = f'ui.boxes.{filename[:-3]}'
                try:
                    # 动态导入模块
                    module = importlib.import_module(module_name)
                    # 遍历模块中的所有类
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # 检查是否是 BaseBox 的子类，并且不是 BaseBox 本身
                        if (issubclass(obj, BaseBox) and
                                obj is not BaseBox and
                                obj not in self.boxes.values()):
                            # 存储 Box 类型
                            self.boxes[name] = obj
                except ImportError as e:
                    print(f"Error importing module {module_name}: {e}")

    def reload_boxes(self):
        # 清空 boxes 字典
        self.boxes.clear()
        # 重新加载 boxes
        self.load_boxes()



if __name__ == '__main__':
    dl = DynamicLoader()
    print(dl.boxes)
