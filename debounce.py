"""Frame-based debouncing for noisy per-frame gesture detections."""


class GestureDebouncer:
    """Requires a gesture to be detected for N consecutive frames before
    reporting it as 'active', and N consecutive frames of absence before
    reporting it as 'inactive'. Returns rising/falling edges.
    """

    def __init__(self, hold_frames: int = 5):
        self.hold_frames = hold_frames
        self.count = 0          # consecutive frames in the candidate state
        self.active = False     # confirmed/stable state
        self.candidate = False  # state we're currently counting toward

    def update(self, raw_detected: bool):
        if raw_detected == self.candidate:
            self.count += 1
        else:
            # detection flipped mid-count -> restart counting the new state
            self.candidate = raw_detected
            self.count = 1

        was_active = self.active
        if self.count >= self.hold_frames:
            self.active = self.candidate

        rising = self.active and not was_active
        falling = (not self.active) and was_active
        return self.active, rising, falling