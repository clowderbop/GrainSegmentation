"""Read YOLO segmentation-format label rows (*.txt polygons in normalized coords)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_yolo_seg_label_rows(
    label_path: Path, *, image_width: int, image_height: int
) -> list[tuple[int, np.ndarray]]:
    rows: list[tuple[int, np.ndarray]] = []
    with label_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            values = [float(value) for value in line.split()]
            if len(values) < 7 or (len(values) - 1) % 2 != 0:
                raise ValueError(f"Invalid segmentation label row in {label_path}")
            class_id = int(values[0])
            points = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= float(image_width)
            points[:, 1] *= float(image_height)
            rows.append((class_id, points))
    return rows
