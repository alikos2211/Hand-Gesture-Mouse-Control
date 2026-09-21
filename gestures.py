"""Pure functions that classify a gesture from a single frame's landmarks."""

import numpy as np


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def is_pinching(landmarks_px: np.ndarray, threshold: float) -> bool:
    """landmarks_px: (21, 3) array in pixel coords."""
    thumb_tip = landmarks_px[4][:2]
    index_tip = landmarks_px[8][:2]
    wrist = landmarks_px[0][:2]
    middle_mcp = landmarks_px[9][:2]

    pinch_dist = _dist(thumb_tip, index_tip)
    hand_scale = _dist(wrist, middle_mcp)  # roughly constant regardless of distance/pose

    if hand_scale < 1e-6:
        return False
    return (pinch_dist / hand_scale) < threshold


def is_rock(landmarks_norm: np.ndarray, threshold: float) -> bool:
    """landmarks_norm: (21, 3) array normalized to [0, 1]."""
    thumb_tip = landmarks_norm[4]
    ring_dip = landmarks_norm[15]
    middle_dip = landmarks_norm[11]
    index_mcp = landmarks_norm[5]
    index_tip = landmarks_norm[8]
    pinky_mcp = landmarks_norm[17]
    pinky_tip = landmarks_norm[20]

    dist_rock = (
        _dist(ring_dip, thumb_tip)
        + _dist(middle_dip, thumb_tip)
        + _dist(middle_dip, ring_dip)
    )
    dist_others_1 = _dist(pinky_tip, index_tip)
    dist_others_2 = _dist(pinky_mcp, index_mcp)

    return dist_rock < threshold and dist_others_1 > dist_others_2


def is_all_fingers_together(landmarks_px: np.ndarray, threshold: float) -> bool:
    """landmarks_px: (21, 3) array in pixel coords."""
    wrist = landmarks_px[0]
    middle_mcp = landmarks_px[9]

    hand_scale = _dist(wrist, middle_mcp)
    if hand_scale < 1e-6:
        return False

    tip_indices = [4, 8, 12, 16, 20]
    tips = [landmarks_px[i][:2] for i in tip_indices]
    centroid = np.mean(tips, axis=0)

    avg_spread = np.mean([_dist(tip, centroid) for tip in tips]) / hand_scale
    return avg_spread < threshold