from dataclasses import dataclass


@dataclass(frozen=True)
class CursorConfig:
    smoothing: float = 0.7
    x_min: float = 0.2
    x_max: float = 0.8
    y_min: float = 0.2
    y_max: float = 0.8


@dataclass(frozen=True)
class GestureConfig:
    pinch_threshold: float = 0.4
    rock_threshold: float = 0.2
    fingers_together_threshold: float = 0.35


@dataclass(frozen=True)
class DebounceConfig:
    pinch_hold_frames: int = 5
    rock_hold_frames: int = 5
    keyboard_hold_frames: int = 8
    screenshot_hold_frames: int = 5


@dataclass(frozen=True)
class ModelConfig:
    palm_weights: str = "MediaPipePyTorch/blazepalm.pth"
    palm_anchors: str = "MediaPipePyTorch/anchors_palm.npy"
    hand_weights: str = "MediaPipePyTorch/blazehand_landmark.pth"
    min_score_thresh: float = 0.75


@dataclass(frozen=True)
class AppConfig:
    cursor: CursorConfig = CursorConfig()
    gesture: GestureConfig = GestureConfig()
    debounce: DebounceConfig = DebounceConfig()
    model: ModelConfig = ModelConfig()
    output_video_path: str = "output.mp4"
    music_file: str = "music.mp3"
    screenshot_path: str = "window.png"
    camera_window_title: str = "Camera"
    quit_key: str = "q"


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                # palm base
]