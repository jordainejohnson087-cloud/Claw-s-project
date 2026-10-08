import math

import cv2
import numpy as np

import console_io
import meter_detector as md


def make_gauge(value, size=400):
    img = np.full((size, size, 3), 255, np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), 180, (0, 0, 0), 4)
    ang = math.radians(225 + value / 100 * 270)
    cv2.line(img, (c, c), (int(c + 150 * math.sin(ang)), int(c - 150 * math.cos(ang))), (0, 0, 0), 5)
    return img


def test_detect():
    for v in (10, 50, 80):
        assert abs(md.detect(make_gauge(v))["value"] - v) < 3


def test_console(tmp_path, capsys):
    cfg = str(tmp_path / "c.json")
    cv2.imwrite(str(tmp_path / "g.png"), make_gauge(50))
    assert console_io.main(["set", "units=psi", "--config", cfg]) == 0
    assert console_io.main(["read", str(tmp_path / "g.png"), "--config", cfg]) == 0
    assert capsys.readouterr().out.strip().endswith("psi")
    assert console_io.main(["set", "bogus=1", "--config", cfg]) == 1
