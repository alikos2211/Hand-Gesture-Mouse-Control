"""Overlay drawing for hand landmarks on a BGR frame."""

from typing import Optional

import cv2
import numpy as np

from config import HAND_CONNECTIONS


def draw_landmarks(frame: np.ndarray, landmarks_px: Optional[np.ndarray]) -> np.ndarray:
    if landmarks_px is None:
        return frame

    pts = [(int(x), int(y)) for x, y in landmarks_px[:, :2]]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return frame