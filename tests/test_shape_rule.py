"""
Checks the rule-based face shape fallback (face_shape_model.classify_face_shape)
with hand-made landmarks whose ratios land in each branch.
Run: python tests/test_shape_rule.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # project root
from types import SimpleNamespace as P

from app.face_shape_model import classify_face_shape


def fake_face(height, forehead, jaw):
    """Landmarks on a 1x1 frame: cheek width is always 1, so the ratios are exactly height/forehead/jaw."""
    pts = [P(x=0.5, y=0.5) for _ in range(478)]
    pts[234], pts[454] = P(x=0.0, y=0.5), P(x=1.0, y=0.5)                      # cheeks
    pts[10],  pts[152] = P(x=0.5, y=0.0), P(x=0.5, y=height)                   # forehead top, chin
    pts[70],  pts[300] = P(x=0.5 - forehead / 2, y=0.2), P(x=0.5 + forehead / 2, y=0.2)
    pts[58],  pts[288] = P(x=0.5 - jaw / 2, y=height - 0.2), P(x=0.5 + jaw / 2, y=height - 0.2)
    return pts


def test_each_shape():
    cases = {
        "Oblong": (1.70, 0.80, 0.80),   # very long face
        "Square": (1.40, 0.95, 0.90),   # wide forehead and jaw
        "Heart":  (1.40, 0.90, 0.70),   # forehead much wider than jaw
        "Oval":   (1.40, 0.80, 0.80),   # balanced
        "Round":  (1.00, 0.80, 0.80),   # as long as it is wide
    }
    for expected, args in cases.items():
        got = classify_face_shape(fake_face(*args), 1, 1)
        assert got == expected, f"{args}: expected {expected}, got {got}"


def test_zero_width_face_returns_none():
    pts = fake_face(1.4, 0.8, 0.8)
    pts[454] = pts[234]
    assert classify_face_shape(pts, 1, 1) is None


if __name__ == '__main__':
    test_each_shape()
    test_zero_width_face_returns_none()
    print("test_shape_rule: all passed")
