# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Live Demo for Online TAPIR."""

import time

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from tapnet.models import tapir_model
from tapnet.utils import model_utils


NUM_POINTS = 20


# ==============================================================================
# Model Loading and Initialization
# ==============================================================================
class TapirModelHandler:
    """Handles TAPIR model loading, initialization, and prediction."""

    def __init__(self, checkpoint_path):
        self.params, self.state = self._load_checkpoint(checkpoint_path)
        self.tapir = tapir_model.ParameterizedTAPIR(
            params=self.params,
            state=self.state,
            tapir_kwargs=dict(
                use_causal_conv=True,
                bilinear_interp_with_depthwise_conv=False,
            ),
        )
        self.online_init_apply = jax.jit(self._online_model_init)
        self.online_predict_apply = jax.jit(self._online_model_predict)
        self.query_features = None
        self.causal_state = None

    def _load_checkpoint(self, checkpoint_path):
        ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
        return ckpt_state["params"], ckpt_state["state"]

    def _online_model_init(self, frames, points):
        feature_grids = self.tapir.get_feature_grids(frames, is_training=False)
        features = self.tapir.get_query_features(
            frames,
            is_training=False,
            query_points=points,
            feature_grids=feature_grids,
        )
        return features

    def _online_model_predict(self, frames, features, causal_context):
        """Compute point tracks and occlusions given frames and query points."""
        feature_grids = self.tapir.get_feature_grids(frames, is_training=False)
        trajectories = self.tapir.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )
        causal_context = trajectories["causal_context"]
        del trajectories["causal_context"]
        return {k: v[-1] for k, v in trajectories.items()}, causal_context

    def initialize(self, frame):
        """Initializes the model with a dummy frame to trigger JIT compilation."""
        query_points = jnp.zeros([NUM_POINTS, 3], dtype=jnp.float32)
        _ = self.online_init_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            points=query_points[None, 0:1],
        )
        jax.block_until_ready(_)  # Wait for compilation to finish

        self.query_features = self.online_init_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            points=query_points[None, :],
        )

        self.causal_state = self.tapir.construct_initial_causal_state(
            NUM_POINTS, len(self.query_features.resolutions) - 1
        )

        prediction, self.causal_state = self.online_predict_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            features=self.query_features,
            causal_context=self.causal_state,
        )

        jax.block_until_ready(prediction["tracks"])  # Wait for compilation

        return self.query_features, self.causal_state

    def predict(self, frame):
        """Predicts tracks and occlusions for a given frame."""
        prediction, self.causal_state = self.online_predict_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            features=self.query_features,
            causal_context=self.causal_state,
        )
        return prediction

    def update_query_features(self, frame, pos, idx_to_update):
        """Updates query features with new point."""
        query_points = jnp.array((0,) + pos, dtype=jnp.float32)
        init_query_features = self.online_init_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            points=query_points[None, None],
        )
        self.query_features, self.causal_state = self.tapir.update_query_features(
            query_features=self.query_features,
            new_query_features=init_query_features,
            idx_to_update=np.array([idx_to_update]),
            causal_state=self.causal_state,
        )

    def track_point(self, frame, pos, idx_to_update):
        """Tracks a new point and updates the model state."""
        self.update_query_features(frame, pos, idx_to_update)

    def update_query_features_batch(self, frame, positions, indices_to_update):
        """批量更新查询特征以提高效率"""
        if not positions or not indices_to_update:
            return

        # 准备批量查询点
        query_points = jnp.array([(0,) + pos for pos in positions], dtype=jnp.float32)
        
        # 初始化批量查询特征
        init_query_features = self.online_init_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            points=query_points[None, :],
        )

        # 使用向量化操作批量更新特征
        self.query_features, self.causal_state = self.tapir.update_query_features(
            query_features=self.query_features,
            new_query_features=init_query_features,
            idx_to_update=np.array(indices_to_update),
            causal_state=self.causal_state,
        )
    def track_points(self, frame, positions, indices_to_update):
        """批量跟踪多个新点并更新模型状态"""
        # 输入验证
        if len(positions) != len(indices_to_update):
            raise ValueError("Positions and indices must be of the same length")
        if max(indices_to_update) >= NUM_POINTS:
            raise ValueError(f"Indices exceed maximum allowed points ({NUM_POINTS-1})")

        # 优化后的批量更新方法
        self.update_query_features_batch(frame, positions, indices_to_update)


# ==============================================================================
# Camera Handling
# ==============================================================================
class CameraHandler:
    """Handles camera initialization and frame retrieval."""

    def __init__(self, camera_id=0, frame_height=240):
        self.vc = cv2.VideoCapture(camera_id)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        if not self.vc.isOpened():
            raise ValueError("Unable to open camera.")
        self.rval, self.frame = self.get_frame()
        if not self.rval:
            raise ValueError("Unable to read from camera.")

    def get_frame(self):
        r_val, image = self.vc.read()
        if not r_val:
            return False, None  # Indicate failure to read frame

        trunc = np.abs(image.shape[1] - image.shape[0]) // 2
        if image.shape[1] > image.shape[0]:
            image = image[:, trunc:-trunc]
        elif image.shape[1] < image.shape[0]:
            image = image[trunc:-trunc]
        return r_val, image

    def release(self):
        self.vc.release()


# ==============================================================================
# User Interface and Visualization
# ==============================================================================
class UIHandler:
    """Handles user interaction and visualization."""

    def __init__(
        self, window_name="Point Tracking", initial_frame=None, tapir_model_handler=None
    ):
        self.window_name = window_name
        self.pos = [None] * NUM_POINTS  # Store positions for each point
        self.query_frame = False
        self.last_click_time = 0
        self.frame = initial_frame  # Store the current frame
        self.have_point = [False] * NUM_POINTS
        self.next_query_idx = 0
        self.tapir_model_handler = tapir_model_handler
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_click)

    def mouse_click(self, event, x, y, flags, param):
        del flags, param
        if (time.time() - self.last_click_time) < 0.5:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.frame is not None:  # Check if frame is available
                # Find the first available slot to store the new point
                for i in range(NUM_POINTS):
                    if not self.have_point[i]:
                        self.pos[i] = (y, self.frame.shape[1] - x)
                        self.next_query_idx = i
                        self.tapir_model_handler.track_point(
                            self.frame, self.pos[i], self.next_query_idx
                        )
                        self.have_point[i] = True
                        self.last_click_time = time.time()
                        return  # Exit after finding an empty slot
                print("All points are being tracked.  Cannot add more.")

    def visualize_tracks(self, frame, prediction):
        """Visualizes the tracked points on the frame."""
        track = prediction["tracks"][0, :, 0]
        occlusion = prediction["occlusion"][0, :, 0]
        expected_dist = prediction["expected_dist"][0, :, 0]
        visibles = model_utils.postprocess_occlusions(occlusion, expected_dist)
        track = np.round(track)

        for i, _ in enumerate(self.have_point):
            if visibles[i] and self.have_point[i]:
                cv2.circle(
                    frame, (int(track[i, 0]), int(track[i, 1])), 5, (255, 0, 0), -1
                )

    def show_frame(self, frame):
        self.frame = frame  # Store the current frame
        cv2.imshow(self.window_name, frame[:, ::-1])

    def get_user_input(self):
        return cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyWindow(self.window_name)


# ==============================================================================
# Main Function
# ==============================================================================
def main():
    print("Welcome to the TAPIR live demo.")
    print("Please note that if the framerate is low (<~12 fps), TAPIR performance")
    print("may degrade and you may need a more powerful GPU.")

    print("Loading model...")
    tapir_model_handler = TapirModelHandler("ui/boxes/TrackingSam/static/causal_tapir_checkpoint.npy")

    print("Initializing camera...")
    camera_handler = CameraHandler()
    frame = camera_handler.frame

    print("Creating UI...")
    ui_handler = UIHandler(
        initial_frame=frame, tapir_model_handler=tapir_model_handler
    )  # Pass initial frame to UI

    print("Compiling jax functions (this may take a while...)")
    tapir_model_handler.initialize(frame)

    t = time.time()
    step_counter = 0

    print("Press ESC to exit.")

    while camera_handler.rval:
        camera_handler.rval, frame = camera_handler.get_frame()
        if not camera_handler.rval:
            print("Error reading from camera. Exiting.")
            break

        # Predict and visualize tracks for all points
        prediction = tapir_model_handler.predict(frame)
        ui_handler.visualize_tracks(frame, prediction)

        ui_handler.show_frame(frame)

        step_counter += 1
        if time.time() - t > 5:
            print(f"{step_counter/(time.time()-t)} frames per second")
            t = time.time()
            step_counter = 0

        key = ui_handler.get_user_input()
        if key == 27:  # exit on ESC
            break

    ui_handler.cleanup()
    camera_handler.release()


if __name__ == "__main__":
    main()
