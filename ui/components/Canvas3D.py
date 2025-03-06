import dearpygui.dearpygui as dpg
import pygfx as gfx
from ui.components.Canvas2D import Canvas2D
import numpy as np
import pylinalg as la
from wgpu.gui.offscreen import WgpuCanvas


class GfxEngine:
    def __init__(self, size=(800, 600), pixel_ratio=1, max_fps=999):
        self.canvas = WgpuCanvas(
            size=size,
            pixel_ratio=pixel_ratio,
            max_fps=max_fps,
        )
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.viewport = gfx.Viewport.from_viewport_or_renderer(self.renderer)
    def draw(self):
        image = np.array(self.canvas.draw())
        return image
    

_engine = GfxEngine()

class BaseScene(gfx.Scene):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale
        # self.world.rotation = la.quat_from_euler((-math.pi / 2, 0, -math.pi / 2))
        self.light = gfx.AmbientLight("#ffffff", 3)
        self.add(self.light)
        self.grid = gfx.GridHelper(100000 * self.scale, 300, color1="#444444", color2="#222222")
        self.grid.local.up = (0, 0, 1)
        self.add(self.grid)
        self.direct_light2 = gfx.DirectionalLight("#ffffff", 1)
        self.direct_light2.local.x = -100
        self.direct_light2.local.y = -200
        self.add(self.direct_light2)
        path = [[0, 0, 0], [0, 0, 0]]
        geometry = gfx.Geometry(positions=path)
        material = gfx.LineMaterial(thickness=10.0, color=(0.8, 0.7, 0.0, 1.0))
        self.test_line = gfx.Line(geometry, material)
        self.add(self.test_line)

class World:
    def __init__(self, SIZE=(800, 600), scale=1):

        self.scale = scale
        self.gfx_engine = GfxEngine(size=SIZE)
        self._canvas = self.gfx_engine.canvas
        self._renderer = self.gfx_engine.renderer
        self._viewport = self.gfx_engine.viewport
        self._scene = BaseScene(scale=self.scale)
        self._scene.add(gfx.AmbientLight())
        directional_light = gfx.DirectionalLight()
        directional_light.world.z = 1
        self._scene.add(directional_light)
        self.mouse_world_position = [0, 0, 0]
        self._camera = gfx.PerspectiveCamera(fov=70, aspect=16 / 9)
        self._camera.world.z = 400 * self.scale
        self.size = SIZE
        self._controller = gfx.OrbitController(self._camera)
        self._canvas.request_draw(lambda: self._renderer.render(self._scene, self._camera))

    def update(self):
        self.handle_event(
            gfx.objects.Event(
                **{
                    "type": "before_render",
                }
            )
        )
        value = (self.gfx_engine.draw()).ravel().astype(np.float32) / 255.0
        return value

    def handle_event(self, event: gfx.objects.Event):
        mouse_x, mouse_y = dpg.get_drawing_mouse_pos()
        screen_width, screen_height = self._canvas.get_logical_size()
        self.mouse_world_position = self.mouse_to_world(mouse_x, mouse_y, screen_width, screen_height)
        self._controller.handle_event(event, self._viewport)

    def mouse_to_world(self, mouse_x, mouse_y, screen_width, screen_height):
        pos = dpg.get_drawing_mouse_pos()
        vs = self._renderer.logical_size
        x = pos[0] / vs[0] * 2 - 1  # 将 x 坐标映射到 [-1, 1]
        y = -(pos[1] / vs[1] * 2 - 1)  # 将 y 坐标映射到 [-1, 1]，并翻转 y 轴
        pos_ndc = (x, y, 0)  # 这里 z 值不再包含在 pos_ndc 中
        projection_matrix = self._camera.projection_matrix 
        view_matrix = self._camera.view_matrix
        ray_origin, ray_direction = self.ndc_to_ray(pos_ndc, projection_matrix, view_matrix)
        intersection = self.ray_plane_intersection(ray_origin, ray_direction)
        return intersection

    def ndc_to_ray(self, ndc_coords, projection_matrix, view_matrix):
        # 逆变换：从NDC到相机坐标
        inverse_projection = np.linalg.inv(projection_matrix)
        camera_coords = inverse_projection @ np.append(ndc_coords, 1.0)
        camera_coords /= camera_coords[3]  # 齐次坐标归一化
        # 逆变换：从相机坐标到世界坐标
        inverse_view = np.linalg.inv(view_matrix)
        world_coords = inverse_view @ np.append(camera_coords[:3], 1.0)
        # 构造射线
        ray_origin = inverse_view[:3, 3]  # 相机位置
        ray_direction = world_coords[:3] - ray_origin
        ray_direction /= np.linalg.norm(ray_direction)  # 归一化方向
        return ray_origin, ray_direction

    def ray_plane_intersection(self, ray_origin, ray_direction, plane_z=0):

        O_x, O_y, O_z = ray_origin
        D_x, D_y, D_z = ray_direction

        # 检查 D_z 是否为 0（射线平行于平面）
        if abs(D_z) < 1e-6:
            return None  # 射线与平面平行，无交点

        # 计算 t
        t = -(O_z - plane_z) / D_z
        # 计算交点坐标
        intersection = ray_origin + t * ray_direction
        intersection = [intersection[0], intersection[1] , plane_z]
        intersection = np.array([intersection[:3]], dtype=np.float32)
        return intersection


class Handler:
    def __init__(self, SIZE=(800, 600), scale=1):
        self.scale = scale
        self.pressedDown = {}
        self.world = World(SIZE, self.scale)
        self.BUTTON_MAP = {
            0: 1,  # MOUSE_LEFT
            1: 2,  # MOUSE_RIGHT
            2: 3,  # MOUSE_MIDDLE
        }
        self.KEY_MAP = {
            568: "w",
            546: "a",
            564: "s",
            549: "d",
        }
        self.WHEEL_RATIO = 500 - (1 - self.scale) * 350

    def callback(self, sender, app_data, user_data):
        try:
            if not dpg.is_item_focused(user_data):
                return
        except:
            pass
        info = dpg.get_item_info(sender)
        type = info["type"]
        if type == "mvAppItemType::mvMouseDragHandler":
            self.world.handle_event(
                gfx.objects.PointerEvent(
                    **{
                        "type": gfx.objects.EventType.POINTER_MOVE,
                        "x": dpg.get_drawing_mouse_pos()[0],
                        "y": dpg.get_drawing_mouse_pos()[1],
                        "button": self.BUTTON_MAP[app_data[0]],
                    }
                )
            )
        elif type == "mvAppItemType::mvMouseClickHandler":
            self.world.handle_event(
                gfx.objects.PointerEvent(
                    **{
                        "type": gfx.objects.EventType.POINTER_DOWN,
                        "x": dpg.get_drawing_mouse_pos()[0],
                        "y": dpg.get_drawing_mouse_pos()[1],
                        "button": self.BUTTON_MAP[app_data],
                    }
                )
            )
        elif type == "mvAppItemType::mvMouseReleaseHandler":
            self.world.handle_event(
                gfx.objects.PointerEvent(
                    **{
                        "type": gfx.objects.EventType.POINTER_UP,
                        "x": dpg.get_drawing_mouse_pos()[0],
                        "y": dpg.get_drawing_mouse_pos()[1],
                        "button": self.BUTTON_MAP[app_data],
                    }
                )
            )
        elif type == "mvAppItemType::mvMouseWheelHandler":
            self.world.handle_event(
                gfx.objects.WheelEvent(
                    **{
                        "type": gfx.objects.EventType.WHEEL,
                        "x": dpg.get_drawing_mouse_pos()[0],
                        "y": dpg.get_drawing_mouse_pos()[1],
                        "dx": -app_data * self.WHEEL_RATIO,
                        "dy": 0,
                    }
                )
            )

class Canvas3D:
    def __init__(
        self,
        parent,
        SIZE=(800, 600),
        scale=1,
        is_mouse_controller=True,
    ):
        self.scale = scale
        self.size = SIZE
        self.is_mouse_controller = is_mouse_controller
        self.canvas = Canvas2D(parent=parent, auto_mouse_transfrom=False)
        self.handler = Handler(SIZE, self.scale)

        self.tag = self.canvas.group_tag
        self.handler_registry()
        self.texture_data = self.texture_registry(SIZE)
        self.camera = self.handler.world._camera
        self.viewport = self.handler.world._viewport
        with self.canvas.draw():
            dpg.draw_image(self.texture_data, pmin=[0, 0], pmax=SIZE)

    def handler_registry(self):
        if not self.is_mouse_controller:
            return
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.handler.callback, user_data=self.canvas.drawlist_tag)
            dpg.add_mouse_release_handler(callback=self.handler.callback, user_data=self.canvas.drawlist_tag)
            dpg.add_mouse_click_handler(callback=self.handler.callback, user_data=self.canvas.drawlist_tag)
            dpg.add_mouse_drag_handler(callback=self.handler.callback, user_data=self.canvas.drawlist_tag)
            # dpg.add_key_press_handler(callback=self.handler.callback, user_data=self.canvas.drawlist_tag)

    def add(self, obj):
        self.handler.world._scene.add(obj)

    def remove(self, obj):
        self.handler.world._scene.remove(obj)

    def texture_registry(self, size):
        width, height = self.size
        texture = np.zeros((size[1], size[0], 4), dtype=np.float32).reshape(-1)
        with dpg.texture_registry():
            texture_tag = dpg.add_raw_texture(
                width=width, height=height, default_value=texture, format=dpg.mvFormat_Float_rgba
            )
        return texture_tag

    def draw(self):
        pass

    def get_world_position(self):
        return self.handler.world.mouse_world_position[0]

    def update(self):
        value = self.handler.world.update()
        dpg.set_value(self.texture_data, value)









## Fixing the OrbitController Rotation
def _update_rotate(self, delta):
    assert isinstance(delta, tuple) and len(delta) == 2
    delta_azimuth, delta_elevation = delta
    camera_state = self._get_camera_state()

    # Note: this code does not use la.vec_euclidean_to_spherical and
    # la.vec_spherical_to_euclidean, because those functions currently
    # have no way to specify a different up vector.

    position = camera_state["position"]
    rotation = camera_state["rotation"]
    up = camera_state["reference_up"]

    # Where is the camera looking at right now
    forward = la.vec_transform_quat((0, 0, -1), rotation)

    # # Get a reference vector, that is orthogonal to up, in a deterministic way.
    # # Might need this if we ever want the azimuth
    # aligned_up = _get_axis_aligned_up_vector(up)
    # orthogonal_vec = np.cross(up, np.roll(aligned_up, 1))

    # Get current elevation, so we can clip it.
    # We don't need the azimuth. When we do, it'd need more care to get a proper 0..2pi range
    elevation = la.vec_angle(forward, up) - 0.5 * np.pi

    # Apply boundaries to the elevation
    new_elevation = elevation + delta_elevation
    bounds = -89 * np.pi / 180, 89 * np.pi / 180
    if new_elevation < bounds[0]:
        delta_elevation = bounds[0] - elevation
    elif new_elevation > bounds[1]:
        delta_elevation = bounds[1] - elevation

    # r_azimuth = la.quat_from_axis_angle(up, -delta_azimuth)
    r_azimuth = la.quat_from_axis_angle((0, 0,np.pi / 2), -delta_azimuth)

    r_elevation = la.quat_from_axis_angle((1, 0, 0), -delta_elevation)

    # Get rotations
    rot1 = rotation


    rot2 = la.quat_mul(r_azimuth, la.quat_mul(rot1, r_elevation))

    # Calculate new position
    pos1 = position
    pos2target1 = self._get_target_vec(camera_state, rotation=rot1)
    pos2target2 = self._get_target_vec(camera_state, rotation=rot2)
    pos2 = pos1 + pos2target1 - pos2target2

    # Apply new state
    new_camera_state = {"position": pos2, "rotation": rot2}
    self._set_camera_state(new_camera_state)
gfx.OrbitController._update_rotate = _update_rotate
