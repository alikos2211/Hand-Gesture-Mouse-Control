"""Maps normalized fingertip coordinates to smoothed screen-cursor movement."""

import pyautogui

from config import CursorConfig


class CursorController:
    def __init__(self, config: CursorConfig, screen_w: int, screen_h: int):
        self.config = config
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.prev_x = screen_w // 2
        self.prev_y = screen_h // 2

    def move_to(self, norm_x: float, norm_y: float) -> None:
        """norm_x, norm_y: index-fingertip coords normalized to [0, 1] over the frame."""
        cfg = self.config

        clamped_x = min(max(norm_x, cfg.x_min), cfg.x_max)
        clamped_y = min(max(norm_y, cfg.y_min), cfg.y_max)

        nx = (clamped_x - cfg.x_min) / (cfg.x_max - cfg.x_min)
        ny = (clamped_y - cfg.y_min) / (cfg.y_max - cfg.y_min)

        target_x = nx * self.screen_w
        target_y = ny * self.screen_h

        curr_x = self.prev_x + (target_x - self.prev_x) * (1 - cfg.smoothing)
        curr_y = self.prev_y + (target_y - self.prev_y) * (1 - cfg.smoothing)

        pyautogui.moveTo(curr_x, curr_y)
        self.prev_x, self.prev_y = curr_x, curr_y