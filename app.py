"""Main loop: camera -> hand detection -> gestures -> cursor/actions -> display."""

import cv2
import numpy as np
import pyautogui

from config import AppConfig
from cursor import CursorController
from debounce import GestureDebouncer
from drawing import draw_landmarks
from hand_model import HandTracker
from actions import ActionController
import gestures


class HandControlApp:
    def __init__(self, config: AppConfig):
        self.config = config

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        screen_w, screen_h = pyautogui.size()

        self.tracker = HandTracker(config.model)
        self.cursor = CursorController(config.cursor, screen_w, screen_h)
        self.actions = ActionController(
            music_file=config.music_file,
            screenshot_path=config.screenshot_path,
            camera_window_title=config.camera_window_title,
        )

        self.pinch_debouncer = GestureDebouncer(config.debounce.pinch_hold_frames)
        self.rock_debouncer = GestureDebouncer(config.debounce.rock_hold_frames)
        self.keyboard_debouncer = GestureDebouncer(config.debounce.keyboard_hold_frames)
        self.screenshot_debouncer = GestureDebouncer(config.debounce.screenshot_hold_frames)

        self.cam = cv2.VideoCapture(0)
        frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cam.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            config.output_video_path, fourcc, fps, (frame_width, frame_height)
        )

    def _handle_detection(self, lm_px: np.ndarray, lm_norm: np.ndarray) -> None:
        cfg = self.config.gesture

        self.cursor.move_to(lm_norm[8][0], lm_norm[8][1])

        _, pinch_rising, _ = self.pinch_debouncer.update(
            gestures.is_pinching(lm_px, cfg.pinch_threshold)
        )
        if pinch_rising:
            pyautogui.click()

        _, rock_rising, rock_falling = self.rock_debouncer.update(
            gestures.is_rock(lm_norm, cfg.rock_threshold)
        )
        if rock_rising:
            self.actions.play_music()
        elif rock_falling:
            self.actions.stop_music()

        _, screenshot_rising, _ = self.screenshot_debouncer.update(
            gestures.is_all_fingers_together(lm_px, cfg.fingers_together_threshold)
        )
        if screenshot_rising:
            self.actions.take_screenshot()

    def _handle_no_detection(self) -> None:
        _, _, rock_falling = self.rock_debouncer.update(False)
        if rock_falling:
            self.actions.stop_music()

        _, _, keyboard_falling = self.keyboard_debouncer.update(False)
        if keyboard_falling:
            self.actions.close_onscreen_keyboard()

        self.pinch_debouncer.update(False)
        self.screenshot_debouncer.update(False)

    def run(self) -> None:
        try:
            while True:
                ret, frame_bgr = self.cam.read()
                if not ret:
                    break

                frame_bgr = cv2.flip(frame_bgr, 1)  # mirror so movement feels natural
                frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])

                detection = self.tracker.detect(frame_rgb)

                if detection is not None:
                    self._handle_detection(detection.landmarks_px, detection.landmarks_norm)
                    frame_bgr = draw_landmarks(frame_bgr, detection.landmarks_px)
                else:
                    self._handle_no_detection()

                self.writer.write(frame_bgr)
                cv2.imshow("Camera", frame_bgr)

                if cv2.waitKey(1) == ord(self.config.quit_key):
                    break
        finally:
            self.cam.release()
            self.writer.release()
            cv2.destroyAllWindows()