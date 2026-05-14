import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np

REPO_YOLO = Path(__file__).resolve().parents[1]
if str(REPO_YOLO) not in sys.path:
    sys.path.insert(0, str(REPO_YOLO))


def _reload_evaluate():
    sys.modules.pop("evaluate", None)
    return importlib.import_module("evaluate")


class EvaluateHelpersTests(unittest.TestCase):
    def test_device_for_sahi_maps_int(self) -> None:
        ev = _reload_evaluate()
        self.assertEqual(ev.device_for_sahi(0), "cuda:0")
        self.assertEqual(ev.device_for_sahi(-1), "cpu")

    def test_device_for_sahi_maps_list(self) -> None:
        ev = _reload_evaluate()
        self.assertEqual(ev.device_for_sahi([1, 2]), "cuda:1")

    def test_load_dataset_config_from_yaml(self) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yaml_path = root / "data.yaml"
            yaml_path.write_text(
                "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: grain\n",
                encoding="utf-8",
            )
            (root / "images" / "val").mkdir(parents=True)
            ds_root, cfg = ev.load_dataset_config_from_yaml(yaml_path)
            self.assertEqual(ds_root, root.resolve())
            self.assertEqual(cfg["val"], "images/val")

    def test_collect_yolo_patch_pairs_lists_images_without_polygon_txt(self) -> None:
        from PIL import Image

        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "val").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                root / "images" / "val" / "tile.png"
            )
            yaml_path = root / "d.yaml"
            yaml_path.write_text(f"path: {root}\ntest: images/val\n", encoding="utf-8")
            dataset_root, cfg = ev.load_dataset_config_from_yaml(yaml_path)
            label_dir, paths = ev.collect_yolo_patch_pairs(dataset_root, cfg)
            self.assertTrue(label_dir.is_dir())
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].stem, "tile")

    def test_collect_yolo_patch_pairs_prefers_test_and_warns_if_val_present(self) -> None:
        from contextlib import redirect_stderr
        from io import StringIO
        from PIL import Image

        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("test", "val"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                root / "images" / "test" / "a.png"
            )
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                root / "images" / "val" / "b.png"
            )
            mask_a = np.zeros((4, 4), dtype=np.uint8)
            Image.fromarray(mask_a, mode="L").save(root / "labels" / "test" / "a_labels.png")
            mask_b = np.zeros((4, 4), dtype=np.uint8)
            Image.fromarray(mask_b, mode="L").save(root / "labels" / "val" / "b_labels.png")
            yaml_path = root / "d.yaml"
            yaml_path.write_text(
                f"path: {root}\ntest: images/test\nval: images/val\n",
                encoding="utf-8",
            )
            dataset_root, cfg = ev.load_dataset_config_from_yaml(yaml_path)
            err = StringIO()
            with redirect_stderr(err):
                label_dir, paths = ev.collect_yolo_patch_pairs(dataset_root, cfg)
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].stem, "a")
            self.assertEqual(label_dir.resolve(), (root / "labels" / "test").resolve())


class EvaluateMainTests(unittest.TestCase):
    def test_parse_args_requires_mode_and_weights(self) -> None:
        ev = _reload_evaluate()
        with self.assertRaises(SystemExit):
            ev.parse_args([])

    def test_parse_args_patches_requires_output_json(self) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            y = Path(tmp) / "d.yaml"
            y.write_text("path: .\ntest: images/val\n", encoding="utf-8")
            w = Path(tmp) / "w.pt"
            w.write_bytes(b"")
            with self.assertRaises(SystemExit):
                ev.parse_args(
                    [
                        "--mode",
                        "patches",
                        "--weights",
                        str(w),
                        "--variant",
                        "PPL",
                        "--data",
                        str(y),
                    ]
                )

    @patch("ultralytics.YOLO")
    def test_run_patches_writes_common_schema(self, mock_yolo_cls: MagicMock) -> None:
        import torch
        from PIL import Image

        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "val").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
                root / "images" / "val" / "tile.png"
            )
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 4:12] = 1
            Image.fromarray(mask, mode="L").save(
                root / "labels" / "val" / "tile_labels.png"
            )
            yaml_path = root / "d.yaml"
            yaml_path.write_text(
                f"path: {root}\ntest: images/val\nnames:\n  0: g\n",
                encoding="utf-8",
            )
            (root / "w.pt").write_bytes(b"")

            class FakeMasks:
                def __init__(self) -> None:
                    m = torch.zeros(16, 16)
                    m[4:12, 4:12] = 1.0
                    self.data = m.unsqueeze(0)

                def __len__(self) -> int:
                    return int(self.data.shape[0])

            class FakeBoxes:
                conf = torch.tensor([0.9])

            class FakeResult:
                masks = FakeMasks()
                boxes = FakeBoxes()

            model = MagicMock()
            model.predict.return_value = [FakeResult()]
            mock_yolo_cls.return_value = model

            out = root / "metrics.json"
            args = SimpleNamespace(
                weights=str(root / "w.pt"),
                device="cpu",
                imgsz=16,
                conf=0.25,
                variant="PPL",
                data=None,
                output_json=out,
                run_ultralytics_val=False,
                batch=1,
                project=None,
                name="t",
                workers=0,
                plots=False,
                half=False,
                save_json=False,
                mask_stem_suffix="_labels",
            )
            ev.run_patches(args, yaml_path)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["model_type"], "yolo")
            self.assertEqual(payload["unit"], "patch")
            self.assertEqual(len(payload["samples"]), 1)
            self.assertEqual(payload["samples"][0]["sample_id"], "tile")
            self.assertIn("aji", payload["samples"][0])
            self.assertIn("extras", payload)

    @patch("ultralytics.YOLO")
    def test_run_patches_raises_without_precomputed_mask(
        self, mock_yolo_cls: MagicMock
    ) -> None:
        from PIL import Image

        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "val").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                root / "images" / "val" / "tile.png"
            )
            yaml_path = root / "d.yaml"
            yaml_path.write_text(
                f"path: {root}\ntest: images/val\nnames:\n  0: g\n",
                encoding="utf-8",
            )
            (root / "w.pt").write_bytes(b"")
            mock_yolo_cls.return_value = MagicMock()
            args = SimpleNamespace(
                weights=str(root / "w.pt"),
                device="cpu",
                imgsz=8,
                conf=0.25,
                variant="PPL",
                data=None,
                output_json=root / "out.json",
                run_ultralytics_val=False,
                batch=1,
                project=None,
                name="t",
                workers=0,
                plots=False,
                half=False,
                save_json=False,
                mask_stem_suffix="_labels",
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                ev.run_patches(args, yaml_path)
            self.assertIn("Pre-computed semantic mask", str(ctx.exception))

    @patch("ultralytics.YOLO")
    def test_run_val_forwards_kwargs(self, mock_yolo_cls: MagicMock) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            y = Path(tmp) / "d.yaml"
            y.write_text("path: .\nval: v\n", encoding="utf-8")
            model = MagicMock()
            model.val.return_value = SimpleNamespace()
            mock_yolo_cls.return_value = model
            args = SimpleNamespace(
                weights=str(Path(tmp) / "w.pt"),
                device="0",
                imgsz=640,
                batch=4,
                workers=2,
                plots=False,
                half=True,
                save_json=True,
                project=Path(tmp) / "proj",
                name="ev1",
            )
            Path(args.weights).write_bytes(b"")
            ev.run_val(args, y)
            model.val.assert_called_once()
            call_kw = model.val.call_args.kwargs
            self.assertEqual(call_kw["data"], str(y))
            self.assertEqual(call_kw["imgsz"], 640)
            self.assertEqual(call_kw["split"], "test")
            self.assertTrue(call_kw["save_json"])
            self.assertEqual(call_kw["name"], "ev1")

    @patch("ultralytics.YOLO")
    def test_run_val_writes_metrics_json_under_project_name(
        self, mock_yolo_cls: MagicMock
    ) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            y = root / "d.yaml"
            y.write_text("path: .\nval: v\n", encoding="utf-8")
            model = MagicMock()
            model.val.return_value = SimpleNamespace(
                box=SimpleNamespace(
                    map=np.float32(0.12),
                    map50=0.34,
                    map75=0.56,
                    maps=np.array([0.12]),
                    image_metrics={
                        "img001": {
                            "precision": np.float64(0.7),
                            "recall": 0.8,
                            "F1": 0.746,
                            "TP": 3,
                            "FP": 1,
                            "FN": 2,
                        }
                    },
                ),
                speed={"inference": np.float32(1.5)},
                results_dict={"metrics/mAP50(B)": 0.34},
            )
            mock_yolo_cls.return_value = model
            args = SimpleNamespace(
                weights=str(root / "w.pt"),
                device="0",
                imgsz=640,
                batch=4,
                workers=2,
                plots=False,
                half=False,
                save_json=False,
                project=root / "proj",
                name="test",
            )
            Path(args.weights).write_bytes(b"")

            ev.run_val(args, y)

            metrics_path = root / "proj" / "test" / "metrics.json"
            self.assertTrue(metrics_path.is_file())
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["box"]["map"], 0.12, places=6)
            self.assertAlmostEqual(payload["box"]["map50"], 0.34)
            self.assertEqual(payload["box"]["maps"], [0.12])
            self.assertEqual(payload["box"]["image_metrics"]["img001"]["TP"], 3)
            self.assertAlmostEqual(payload["speed"]["inference"], 1.5)
            self.assertAlmostEqual(payload["results_dict"]["metrics/mAP50(B)"], 0.34)

    @patch("ultralytics.YOLO")
    def test_run_val_forwards_name_without_project(
        self, mock_yolo_cls: MagicMock
    ) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            y = Path(tmp) / "d.yaml"
            y.write_text("path: .\nval: v\n", encoding="utf-8")
            model = MagicMock()
            model.val.return_value = SimpleNamespace()
            mock_yolo_cls.return_value = model
            args = SimpleNamespace(
                weights=str(Path(tmp) / "w.pt"),
                device="0",
                imgsz=640,
                batch=4,
                workers=2,
                plots=False,
                half=True,
                save_json=False,
                project=None,
                name="my_run",
            )
            Path(args.weights).write_bytes(b"")
            ev.run_val(args, y)
            model.val.assert_called_once()
            call_kw = model.val.call_args.kwargs
            self.assertEqual(call_kw["name"], "my_run")
            self.assertNotIn("project", call_kw)

    def test_load_sahi_pairs_manifest_relative_to_manifest_dir(self) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "data"
            sub.mkdir()
            (sub / "a.tif").write_bytes(b"")
            (sub / "a.gpkg").write_bytes(b"")
            manifest = sub / "manifest.json"
            manifest.write_text(
                json.dumps([{"test_tiff": "a.tif", "test_gpkg": "a.gpkg"}]),
                encoding="utf-8",
            )
            old = os.getcwd()
            try:
                os.chdir("/")
                args = SimpleNamespace(
                    manifest=manifest,
                    test_tiff=None,
                    test_gpkg=None,
                )
                pairs = ev._load_sahi_pairs(args)
            finally:
                os.chdir(old)
            self.assertEqual(pairs[0][0], (sub / "a.tif").resolve())
            self.assertEqual(pairs[0][1], (sub / "a.gpkg").resolve())

    @patch("sahi.AutoDetectionModel.from_pretrained")
    def test_run_sahi_creates_parent_dirs_for_output_json(
        self,
        mock_from_pretrained: MagicMock,
    ) -> None:
        ev = _reload_evaluate()
        mock_from_pretrained.return_value = MagicMock()
        fake_img = np.zeros((10, 10, 1), dtype=np.uint8)
        with patch.object(
            ev,
            "_get_sliced_prediction_preserve_channels",
            return_value=MagicMock(object_prediction_list=[]),
        ):
            with patch.object(ev, "load_image_for_yolo", return_value=fake_img):
                with patch.object(ev, "load_polygons_from_gpkg", return_value=[]):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        tiff = root / "tile.tif"
                        gpkg = root / "tile.gpkg"
                        tiff.write_bytes(b"")
                        gpkg.write_bytes(b"")
                        out = root / "nested" / "deep" / "metrics.json"
                        args = SimpleNamespace(
                            weights=str(root / "w.pt"),
                            device="cpu",
                            conf=0.25,
                            slice_height=64,
                            slice_width=64,
                            overlap_height_ratio=0.2,
                            overlap_width_ratio=0.2,
                            test_tiff=tiff,
                            test_gpkg=gpkg,
                            manifest=None,
                            sahi_out_dir=None,
                            output_json=out,
                        )
                        (root / "w.pt").write_bytes(b"")
                        ev.run_sahi(args)
                        self.assertTrue(out.is_file())
                        self.assertTrue(out.parent.is_dir())

    @patch("sahi.AutoDetectionModel.from_pretrained")
    def test_run_sahi_writes_mask_only_visual_and_gpkg_under_out_dir(
        self,
        mock_from_pretrained: MagicMock,
    ) -> None:
        ev = _reload_evaluate()
        mock_from_pretrained.return_value = MagicMock()
        result = MagicMock(object_prediction_list=[])

        def export_visuals(export_dir: str, file_name: str) -> None:
            Path(export_dir, f"{file_name}.png").write_bytes(b"visual")

        result.export_visuals.side_effect = export_visuals
        fake_img = np.zeros((10, 10, 1), dtype=np.uint8)
        with patch.object(
            ev,
            "_get_sliced_prediction_preserve_channels",
            return_value=result,
        ):
            with patch.object(ev, "load_image_for_yolo", return_value=fake_img):
                with patch.object(ev, "load_polygons_from_gpkg", return_value=[]):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        tiff = root / "tile.tif"
                        gpkg = root / "tile.gpkg"
                        tiff.write_bytes(b"")
                        gpkg.write_bytes(b"")
                        out_dir = root / "sahi_out"
                        args = SimpleNamespace(
                            weights=str(root / "w.pt"),
                            device="cpu",
                            conf=0.25,
                            slice_height=64,
                            slice_width=64,
                            overlap_height_ratio=0.2,
                            overlap_width_ratio=0.2,
                            test_tiff=tiff,
                            test_gpkg=gpkg,
                            manifest=None,
                            sahi_out_dir=out_dir,
                            output_json=root / "metrics.json",
                        )
                        (root / "w.pt").write_bytes(b"")
                        ev.run_sahi(args)

                        visual = out_dir / "tile" / "prediction_visual.png"
                        mask = out_dir / "tile" / "predicted_masks.gpkg"
                        self.assertTrue(visual.is_file())
                        self.assertTrue(mask.is_file())
                        self.assertFalse((out_dir / "tile" / "predicted_masks.tif").exists())
                        result.export_visuals.assert_not_called()
                        self.assertEqual(len(gpd.read_file(mask)), 0)

    def test_write_predicted_masks_gpkg_saves_prediction_polygons(self) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "predicted_masks.gpkg"
            ev.write_predicted_masks_gpkg(
                [
                    {
                        "category_id": 1,
                        "score": np.float32(0.9),
                        "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]],
                    }
                ],
                height=10,
                width=10,
                out_path=out,
            )

            gdf = gpd.read_file(out)
            self.assertEqual(len(gdf), 1)
            self.assertEqual(int(gdf.loc[0, "instance_id"]), 1)
            self.assertAlmostEqual(float(gdf.loc[0, "score"]), 0.9, places=6)
            self.assertGreater(gdf.loc[0, "geometry"].area, 0)

    @patch("sahi.predict.get_sliced_prediction")
    @patch("sahi.AutoDetectionModel.from_pretrained")
    def test_run_sahi_always_uses_preserve_channels_slicer(
        self,
        mock_from_pretrained: MagicMock,
        mock_get_sliced_prediction: MagicMock,
    ) -> None:
        """SAHI get_sliced_prediction is unused; all channel layouts use the custom slicer."""
        ev = _reload_evaluate()
        mock_from_pretrained.return_value = MagicMock()
        result = MagicMock(object_prediction_list=[])
        fake_img = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch.object(ev, "load_image_for_yolo", return_value=fake_img):
            with patch.object(ev, "load_polygons_from_gpkg", return_value=[]):
                with patch.object(
                    ev, "_get_sliced_prediction_preserve_channels", return_value=result
                ) as mock_preserve:
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        tiff = root / "tile.tif"
                        gpkg = root / "tile.gpkg"
                        tiff.write_bytes(b"")
                        gpkg.write_bytes(b"")
                        args = SimpleNamespace(
                            weights=str(root / "w.pt"),
                            device="cpu",
                            conf=0.25,
                            slice_height=64,
                            slice_width=64,
                            overlap_height_ratio=0.2,
                            overlap_width_ratio=0.2,
                            test_tiff=tiff,
                            test_gpkg=gpkg,
                            manifest=None,
                            sahi_out_dir=None,
                            output_json=root / "metrics.json",
                        )
                        (root / "w.pt").write_bytes(b"")
                        ev.run_sahi(args)

        mock_get_sliced_prediction.assert_not_called()
        mock_preserve.assert_called_once()

    def test_aggregate_sahi_means_excludes_undefined(self) -> None:
        ev = _reload_evaluate()
        row_ok = {
            "AP": 0.5,
            "AP50": 0.6,
            "AP75": 0.7,
            "APs": 0.1,
            "APm": 0.2,
            "APl": 0.3,
            "AR1": 0.4,
            "AR10": 0.5,
            "AR100": 0.6,
        }
        row_empty_gt = {
            "AP": -1.0,
            "AP50": -1.0,
            "AP75": -1.0,
            "APs": -1.0,
            "APm": -1.0,
            "APl": -1.0,
            "AR1": -1.0,
            "AR10": -1.0,
            "AR100": -1.0,
        }
        means = ev.aggregate_sahi_means([row_ok, row_empty_gt])
        self.assertAlmostEqual(means["mean_AP"], 0.5)
        self.assertAlmostEqual(means["mean_AP50"], 0.6)

    def test_aggregate_sahi_means_instance_metrics(self) -> None:
        ev = _reload_evaluate()
        row_a = {
            "AP": 0.5,
            "AP50": 0.5,
            "AP75": 0.5,
            "APs": 0.0,
            "APm": 0.0,
            "APl": 0.5,
            "AR1": 0.0,
            "AR10": 0.5,
            "AR100": 0.5,
            "aji": 0.8,
            "precision_iou50": 0.9,
            "recall_iou50": 0.7,
            "f1_iou50": 0.75,
            "precision_iou75": 0.8,
            "recall_iou75": 0.6,
            "f1_iou75": 0.65,
            "mP_iou50_95": 0.85,
            "mR_iou50_95": 0.65,
            "mF1_iou50_95": 0.7,
        }
        row_b = {**row_a, "aji": 0.4, "f1_iou50": 0.5}
        means = ev.aggregate_sahi_means([row_a, row_b])
        self.assertAlmostEqual(means["mean_aji"], 0.6)
        self.assertAlmostEqual(means["mean_f1_iou50"], 0.625)

    def test_aggregate_sahi_means_single_image_all_undefined(self) -> None:
        ev = _reload_evaluate()
        row = {
            "AP": -1.0,
            "AP50": -1.0,
            "AP75": -1.0,
            "APs": -1.0,
            "APm": -1.0,
            "APl": -1.0,
            "AR1": -1.0,
            "AR10": -1.0,
            "AR100": -1.0,
        }
        means = ev.aggregate_sahi_means([row])
        self.assertIsNone(means["mean_AP"])
        self.assertIsNone(means["mean_AP50"])

    @patch("sahi.AutoDetectionModel.from_pretrained")
    def test_run_sahi_output_json_uses_null_not_nan_for_undefined_means(
        self,
        mock_from_pretrained: MagicMock,
    ) -> None:
        """Written metrics.json must be strict JSON: null for undefined mean_*, no NaN."""
        ev = _reload_evaluate()
        mock_from_pretrained.return_value = MagicMock()
        fake_img = np.zeros((10, 10, 1), dtype=np.uint8)
        with patch.object(
            ev,
            "_get_sliced_prediction_preserve_channels",
            return_value=MagicMock(object_prediction_list=[]),
        ):
            with patch.object(ev, "load_image_for_yolo", return_value=fake_img):
                with patch.object(ev, "load_polygons_from_gpkg", return_value=[]):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        tiff = root / "tile.tif"
                        gpkg = root / "tile.gpkg"
                        tiff.write_bytes(b"")
                        gpkg.write_bytes(b"")
                        out = root / "m.json"
                        args = SimpleNamespace(
                            weights=str(root / "w.pt"),
                            device="cpu",
                            conf=0.25,
                            slice_height=64,
                            slice_width=64,
                            overlap_height_ratio=0.2,
                            overlap_width_ratio=0.2,
                            test_tiff=tiff,
                            test_gpkg=gpkg,
                            manifest=None,
                            sahi_out_dir=None,
                            output_json=out,
                        )
                        (root / "w.pt").write_bytes(b"")
                        ev.run_sahi(args)
                        text = out.read_text(encoding="utf-8")
                        self.assertNotIn("NaN", text)
                        self.assertNotIn("Infinity", text)
                        loaded = json.loads(text)
                        self.assertIsNone(loaded["mean_AP"])
                        self.assertIsNone(loaded["mean_AP50"])
                        self.assertAlmostEqual(loaded["mean_aji"], 1.0)
                        self.assertAlmostEqual(loaded["mean_mF1_iou50_95"], 1.0)
                        row0 = loaded["per_image"][0]
                        self.assertTrue(row0["empty_gt"])
                        self.assertAlmostEqual(row0["aji"], 1.0)
                        self.assertAlmostEqual(row0["f1_iou50"], 1.0)


if __name__ == "__main__":
    unittest.main()
