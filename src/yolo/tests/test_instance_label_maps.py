import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

REPO_YOLO = Path(__file__).resolve().parents[1]
_SRC = REPO_YOLO.parent
for _root in (_SRC, REPO_YOLO):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


def _reload_maps():
    sys.modules.pop("instance_label_maps", None)
    return importlib.import_module("instance_label_maps")


def _reload_metrics():
    sys.modules.pop("evaluation.metrics", None)
    sys.modules.pop("evaluation.instance_masks", None)
    return importlib.import_module("evaluation.metrics")


class InstanceLabelMapsCocoStyleTests(unittest.TestCase):
    def test_gt_and_dt_perfect_match_aji_one(self) -> None:
        ilm = _reload_maps()
        metrics = _reload_metrics()
        h, w = 12, 12
        ring = [2.0, 2.0, 9.0, 2.0, 9.0, 9.0, 2.0, 9.0]
        gt_anns = [{"id": 1, "segmentation": [ring]}]
        dt_anns = [{"segmentation": [ring], "score": 0.99}]
        gtm = ilm.gt_annotations_to_instance_map(gt_anns, h, w)
        pdm = ilm.dt_annotations_to_instance_map(dt_anns, h, w)
        self.assertAlmostEqual(metrics.compute_aji(gtm, pdm), 1.0)
        d = metrics.compute_instance_metrics_dict(gtm, pdm)
        self.assertAlmostEqual(d["f1_iou50"], 1.0)
        self.assertAlmostEqual(d["mF1_iou50_95"], 1.0)

    def test_both_empty_maps_metrics_are_perfect(self) -> None:
        ilm = _reload_maps()
        metrics = _reload_metrics()
        h, w = 8, 8
        z = ilm.gt_annotations_to_instance_map([], h, w)
        z2 = ilm.dt_annotations_to_instance_map([], h, w)
        self.assertAlmostEqual(metrics.compute_aji(z, z2), 1.0)
        d = metrics.compute_instance_metrics_dict(z, z2)
        self.assertAlmostEqual(d["f1_iou50"], 1.0)

    def test_empty_gt_nonempty_pred_aji_zero(self) -> None:
        ilm = _reload_maps()
        metrics = _reload_metrics()
        h, w = 10, 10
        ring = [1.0, 1.0, 8.0, 1.0, 8.0, 8.0, 1.0, 8.0]
        gtm = ilm.gt_annotations_to_instance_map([], h, w)
        pdm = ilm.dt_annotations_to_instance_map(
            [{"segmentation": [ring], "score": 0.5}], h, w
        )
        self.assertAlmostEqual(metrics.compute_aji(gtm, pdm), 0.0)
        d = metrics.compute_instance_metrics_dict(gtm, pdm)
        self.assertAlmostEqual(d["f1_iou50"], 0.0)

    def test_nonempty_gt_empty_pred_aji_zero(self) -> None:
        ilm = _reload_maps()
        metrics = _reload_metrics()
        h, w = 10, 10
        ring = [1.0, 1.0, 8.0, 1.0, 8.0, 8.0, 1.0, 8.0]
        gtm = ilm.gt_annotations_to_instance_map(
            [{"id": 1, "segmentation": [ring]}], h, w
        )
        pdm = ilm.dt_annotations_to_instance_map([], h, w)
        self.assertAlmostEqual(metrics.compute_aji(gtm, pdm), 0.0)
        d = metrics.compute_instance_metrics_dict(gtm, pdm)
        self.assertAlmostEqual(d["recall_iou50"], 0.0)

    def test_dt_rle_segmentation_decodes(self) -> None:
        ilm = _reload_maps()
        h, w = 12, 12
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[3:9, 3:9] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        dt_anns = [{"segmentation": rle, "score": 0.95}]
        pdm = ilm.dt_annotations_to_instance_map(dt_anns, h, w)
        self.assertTrue((pdm[mask.astype(bool)] == 1).all())
        self.assertEqual(int(pdm[0, 0]), 0)

    def test_dt_overlap_higher_score_wins_pixels(self) -> None:
        ilm = _reload_maps()
        h, w = 10, 10
        left = [0.0, 0.0, 6.0, 0.0, 6.0, 10.0, 0.0, 10.0]
        right = [4.0, 0.0, 10.0, 0.0, 10.0, 10.0, 4.0, 10.0]
        dt_anns = [
            {"segmentation": [left], "score": 0.1},
            {"segmentation": [right], "score": 0.9},
        ]
        pdm = ilm.dt_annotations_to_instance_map(dt_anns, h, w)
        self.assertEqual(int(pdm[5, 5]), 2)

    def test_gt_sorted_by_id_stable_under_reversed_input_order(self) -> None:
        ilm = _reload_maps()
        h, w = 14, 14
        a = [2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]
        b = [5.0, 5.0, 10.0, 5.0, 10.0, 10.0, 5.0, 10.0]
        g1 = ilm.gt_annotations_to_instance_map(
            [{"id": 1, "segmentation": [a]}, {"id": 2, "segmentation": [b]}], h, w
        )
        g2 = ilm.gt_annotations_to_instance_map(
            [{"id": 2, "segmentation": [b]}, {"id": 1, "segmentation": [a]}], h, w
        )
        np.testing.assert_array_equal(g1, g2)


class YoloLabelMapTests(unittest.TestCase):
    def test_yolo_label_txt_rasterizes_polygon(self) -> None:
        maps = _reload_maps()
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "a.txt"
            label.write_text(
                "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
                encoding="utf-8",
            )
            inst = maps.yolo_seg_label_txt_to_instance_map(label, 10, 10)
            self.assertEqual(int(inst[2:8, 2:8].min()), 1)
            self.assertEqual(int(inst[2:8, 2:8].max()), 1)
            self.assertEqual(len([x for x in np.unique(inst) if x != 0]), 1)

    def test_binary_masks_higher_confidence_overwrites_overlap(self) -> None:
        maps = _reload_maps()
        m = np.zeros((2, 6, 6), dtype=bool)
        m[0, :, :4] = True
        m[1, 3:, 3:] = True
        conf = np.array([0.1, 0.9])
        out = maps.binary_masks_to_instance_map_by_confidence(m, conf)
        self.assertTrue(np.all(out[3:6, 3:4] == 2))


if __name__ == "__main__":
    unittest.main()
