"""Wraps BlazePalm + BlazeHandLandmark model loading and per-frame inference."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from MediaPipePyTorch.blazebase import resize_pad, denormalize_detections
from MediaPipePyTorch.blazepalm import BlazePalm
from MediaPipePyTorch.blazehand_landmark import BlazeHandLandmark

from config import ModelConfig


@dataclass
class HandDetection:
    landmarks_px: np.ndarray     # (21, 3) pixel coords in the input frame
    landmarks_norm: np.ndarray   # (21, 3) normalized to [0, 1] by frame width/height


class HandTracker:
    def __init__(self, config: ModelConfig, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Running hand tracking on:", self.device)
        torch.set_grad_enabled(False)

        self.palm_detector = BlazePalm().to(self.device)
        self.palm_detector.load_weights(config.palm_weights)
        self.palm_detector.load_anchors(config.palm_anchors)
        self.palm_detector.min_score_thresh = config.min_score_thresh

        self.hand_regressor = BlazeHandLandmark().to(self.device)
        self.hand_regressor.load_weights(config.hand_weights)

    def detect(self, frame_rgb: np.ndarray) -> Optional[HandDetection]:
        """Runs palm detection + landmark regression on one RGB frame.

        Returns None if no confident hand is found.
        """
        frame_h, frame_w = frame_rgb.shape[:2]

        img1, img2, scale, pad = resize_pad(frame_rgb)
        normalized_palm_detections = self.palm_detector.predict_on_image(img1)
        palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

        if len(palm_detections) == 0:
            return None

        xc, yc, roi_scale, theta = self.palm_detector.detection2roi(palm_detections.cpu())
        img, affine, box = self.hand_regressor.extract_roi(frame_rgb, xc, yc, theta, roi_scale)
        flags, handed, normalized_landmarks = self.hand_regressor(img.to(self.device))
        landmarks_px_all = self.hand_regressor.denormalize_landmarks(
            normalized_landmarks.cpu(), affine
        )

        best_i = int(torch.argmax(flags))
        if flags[best_i] <= 0.5:
            return None

        landmarks_px = landmarks_px_all[best_i].numpy()  # (21, 3) pixel coords

        landmarks_norm = landmarks_px.copy()
        landmarks_norm[:, 0] /= frame_w
        landmarks_norm[:, 1] /= frame_h

        return HandDetection(landmarks_px=landmarks_px, landmarks_norm=landmarks_norm)