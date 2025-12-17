from __future__ import annotations
import numpy as np


def load_points_txt(path: str, *, drop_zero_rows: bool = True) -> np.ndarray:
    pts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if drop_zero_rows and x == 0.0 and y == 0.0:
                continue
            pts.append((x, y))
    return np.asarray(pts, dtype=np.float64)


def downsample_uniform(points: np.ndarray, step: int = 2) -> np.ndarray:
    """Prende 1 punto ogni 'step' (utile per real-time)."""
    if step <= 1:
        return points
    return points[::step].copy()