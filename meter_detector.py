"""Analog meter (gauge) detection with OpenCV. All parameters live in a JSON config that can be updated."""
import json
import math
import os

import cv2
import numpy as np

DEFAULT_CONFIG = {
    "min_angle": 225.0,   # needle angle (deg, clockwise from 12 o'clock) at min_value
    "max_angle": 135.0,   # needle angle at max_value (sweeps clockwise from min_angle)
    "min_value": 0.0,
    "max_value": 100.0,
    "units": "",
    "blur": 5,            # odd Gaussian kernel size
    "canny_low": 50,
    "canny_high": 150,
    "hough_threshold": 40,
    "min_line_ratio": 0.3,  # min needle length as a ratio of meter radius
    "center_tolerance": 0.2,  # max needle distance from centre as ratio of radius
}


def load_config(path=None):
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def update_config(path, updates):
    """Merge updates into the config file at path; unknown keys are rejected."""
    cfg = load_config(path)
    for key, value in updates.items():
        if key not in DEFAULT_CONFIG:
            raise KeyError(f"unknown config key: {key}")
        default = DEFAULT_CONFIG[key]
        cfg[key] = type(default)(value) if not isinstance(default, str) else str(value)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def find_meter(gray):
    """Return (cx, cy, r) of the dial circle; falls back to the image centre."""
    h, w = gray.shape
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min(h, w) / 2,
                               param1=100, param2=40,
                               minRadius=min(h, w) // 4, maxRadius=min(h, w) // 2)
    if circles is not None:
        cx, cy, r = circles[0][0]
        return float(cx), float(cy), float(r)
    return w / 2.0, h / 2.0, min(h, w) / 2.0


def _angle(cx, cy, x, y):
    """Clockwise angle in degrees from 12 o'clock."""
    return math.degrees(math.atan2(x - cx, cy - y)) % 360


def detect(image, config=None):
    """Detect the needle and return {'value', 'angle', 'center', 'radius'}; raises ValueError if none."""
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    k = int(cfg["blur"]) | 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    cx, cy, r = find_meter(gray)
    edges = cv2.Canny(gray, cfg["canny_low"], cfg["canny_high"])
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, int(cfg["hough_threshold"]),
                            minLineLength=int(r * cfg["min_line_ratio"]), maxLineGap=5)
    best = None
    for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else []):
        # distance of the centre from the line
        length = math.hypot(x2 - x1, y2 - y1)
        dist = abs((x2 - x1) * (y1 - cy) - (x1 - cx) * (y2 - y1)) / length
        if dist > cfg["center_tolerance"] * r:
            continue
        if best is None or length > best[0]:
            far = (x1, y1) if math.hypot(x1 - cx, y1 - cy) > math.hypot(x2 - cx, y2 - cy) else (x2, y2)
            best = (length, far)
    if best is None:
        raise ValueError("no needle detected")
    angle = _angle(cx, cy, *best[1])
    span = (cfg["max_angle"] - cfg["min_angle"]) % 360 or 360.0
    frac = ((angle - cfg["min_angle"]) % 360) / span
    value = cfg["min_value"] + frac * (cfg["max_value"] - cfg["min_value"])
    return {"value": value, "angle": angle, "center": (cx, cy), "radius": r}
