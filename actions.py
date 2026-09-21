"""Side-effecting actions triggered by confirmed gestures."""

import os
import subprocess

import pygame
import pyautogui
import pygetwindow as gw


class ActionController:
    def __init__(self, music_file: str, screenshot_path: str, camera_window_title: str):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.music_path = os.path.join(script_dir, music_file)
        self.screenshot_path = screenshot_path
        self.camera_window_title = camera_window_title
        pygame.init()

    def play_music(self) -> None:
        pygame.mixer.music.load(self.music_path)
        pygame.mixer.music.play()

    def stop_music(self) -> None:
        pygame.mixer.music.stop()

    def take_screenshot(self) -> None:
        windows = gw.getWindowsWithTitle(self.camera_window_title)
        if not windows:
            return
        window = windows[0]
        img = pyautogui.screenshot(
            region=(window.left, window.top, window.width, window.height)
        )
        img.save(self.screenshot_path)

    @staticmethod
    def close_onscreen_keyboard() -> None:
        subprocess.run(["taskkill", "/IM", "osk.exe", "/F"], shell=True)