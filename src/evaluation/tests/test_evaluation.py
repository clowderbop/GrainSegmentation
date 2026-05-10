import contextlib
import importlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


REPO_SRC = Path(__file__).resolve().parents[2]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def _install_tensorflow_stub() -> None:
    tf_module = types.ModuleType("tensorflow")
    tf_module.Tensor = object
    tf_module.keras = SimpleNamespace(
        Model=object,
        models=SimpleNamespace(load_model=lambda *args, **kwargs: None),
    )
    sys.modules["tensorflow"] = tf_module


def _install_evaluate_import_stubs() -> None:
    _install_tensorflow_stub()

    training_pkg = types.ModuleType("training")
    data_module = types.ModuleType("training.data")
    model_module = types.ModuleType("training.model")

    data_module.list_samples = lambda *args, **kwargs: []
    data_module._load_rgb_image = lambda path: np.zeros((2, 2, 3), dtype=np.float32)
    data_module._load_raster_mask = lambda path: np.zeros((2, 2), dtype=np.int32)
    model_module.weighted_crossentropy = lambda y_true, y_pred: 0.0

    training_pkg.data = data_module
    training_pkg.model = model_module

    sys.modules["training"] = training_pkg
    sys.modules["training.data"] = data_module
    sys.modules["training.model"] = model_module


def _reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _bruteforce_instance_iou_matrix(
    true_instances: np.ndarray, pred_instances: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
    """Reference IoU matrix via per-instance boolean masks (slow; tests only)."""
    true_ids = sorted(int(x) for x in np.unique(true_instances) if x != 0)
    pred_ids = sorted(int(x) for x in np.unique(pred_instances) if x != 0)
    nt, np_ = len(true_ids), len(pred_ids)
    mat = np.zeros((nt, np_), dtype=np.float64)
    for i, tid in enumerate(true_ids):
        tm = true_instances == tid
        for j, pid in enumerate(pred_ids):
            pm = pred_instances == pid
            inter = int(np.logical_and(tm, pm).sum())
            union = int(np.logical_or(tm, pm).sum())
            mat[i, j] = float(inter) / float(union) if union > 0 else 0.0
    return mat, true_ids, pred_ids


class MetricsTests(unittest.TestCase):
    def test_compute_aji_penalizes_merged_predictions(self) -> None:
        metrics = _reload_module("evaluation.metrics")

        true_instances = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
            ],
            dtype=np.int32,
        )
        pred_instances = np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
            ],
            dtype=np.int32,
        )

        self.assertAlmostEqual(
            metrics.compute_aji(true_instances, pred_instances),
            1.0 / 3.0,
        )

    def test_instance_prf_perfect_match(self) -> None:
        metrics = _reload_module("evaluation.metrics")
        lab = np.array([[1, 1], [1, 0]], dtype=np.int32)
        p, r, f = metrics.compute_instance_precision_recall_f1(lab, lab, 0.5)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(f, 1.0)

    def test_instance_prf_empty_both(self) -> None:
        metrics = _reload_module("evaluation.metrics")
        empty = np.zeros((4, 4), dtype=np.int32)
        p, r, f = metrics.compute_instance_precision_recall_f1(empty, empty, 0.5)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(f, 1.0)

    def test_instance_prf_empty_gt_with_predictions_is_all_zero(self) -> None:
        metrics = _reload_module("evaluation.metrics")
        gt = np.zeros((8, 8), dtype=np.int32)
        pr = np.zeros((8, 8), dtype=np.int32)
        pr[1:4, 1:4] = 1
        p, r, f = metrics.compute_instance_precision_recall_f1(gt, pr, 0.5)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f, 0.0)

    def test_instance_prf_empty_predictions_with_gt_is_all_zero(self) -> None:
        metrics = _reload_module("evaluation.metrics")
        gt = np.zeros((8, 8), dtype=np.int32)
        gt[1:4, 1:4] = 1
        pr = np.zeros((8, 8), dtype=np.int32)
        p, r, f = metrics.compute_instance_precision_recall_f1(gt, pr, 0.5)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f, 0.0)

    def test_instance_prf_extra_prediction(self) -> None:
        metrics = _reload_module("evaluation.metrics")
        gt = np.zeros((8, 8), dtype=np.int32)
        gt[1:4, 1:4] = 1
        pr = np.zeros((8, 8), dtype=np.int32)
        pr[1:4, 1:4] = 1
        pr[5:7, 5:7] = 2
        p, r, f = metrics.compute_instance_precision_recall_f1(gt, pr, 0.5)
        self.assertAlmostEqual(p, 0.5)
        self.assertEqual(r, 1.0)
        self.assertAlmostEqual(f, 2.0 * 0.5 * 1.0 / 1.5)

    def test_instance_prf_missed_ground_truth(self) -> None:
        metrics = _reload_module("evaluation.metrics")
        gt = np.zeros((8, 8), dtype=np.int32)
        gt[1:4, 1:4] = 1
        gt[5:7, 5:7] = 2
        pr = np.zeros((8, 8), dtype=np.int32)
        pr[1:4, 1:4] = 1
        p, r, f = metrics.compute_instance_precision_recall_f1(gt, pr, 0.5)
        self.assertEqual(p, 1.0)
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(f, 2.0 / 3.0)

    def test_mean_iou_sweep_uses_ten_thresholds(self) -> None:
        metrics = _reload_module("evaluation.metrics")
        self.assertEqual(len(metrics.IOU_THRESHOLDS_50_95), 10)
        self.assertAlmostEqual(metrics.IOU_THRESHOLDS_50_95[0], 0.5)
        self.assertAlmostEqual(metrics.IOU_THRESHOLDS_50_95[-1], 0.95)
        self.assertEqual(metrics._index_for_reported_threshold(0.75), 5)

    def test_build_instance_iou_matrix_matches_bruteforce(self) -> None:
        """Histogram IoU matches per-pair mask IoU (regression guard for build_instance_iou_matrix)."""
        metrics = _reload_module("evaluation.metrics")
        rng = np.random.default_rng(42)
        for trial in range(40):
            h, w = int(rng.integers(3, 14)), int(rng.integers(3, 14))
            gt = rng.integers(0, 7, size=(h, w), dtype=np.int32)
            pr = rng.integers(0, 7, size=(h, w), dtype=np.int32)
            fast, t_ids, p_ids = metrics.build_instance_iou_matrix(gt, pr)
            slow, t2, p2 = _bruteforce_instance_iou_matrix(gt, pr)
            self.assertEqual(t_ids, t2)
            self.assertEqual(p_ids, p2)
            np.testing.assert_allclose(
                fast,
                slow,
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"trial {trial} shape {gt.shape}",
            )

    def test_merge_two_gt_one_pred_no_pair_at_iou50(self) -> None:
        """Merged prediction can fall below 0.5 IoU with each GT; greedy match yields 0 TP."""
        metrics = _reload_module("evaluation.metrics")
        gt = np.zeros((10, 10), dtype=np.int32)
        gt[1:4, 1:4] = 1
        gt[6:9, 6:9] = 2
        pr = np.zeros((10, 10), dtype=np.int32)
        pr[1:9, 1:9] = 1
        p50, r50, _ = metrics.compute_instance_precision_recall_f1(gt, pr, 0.5)
        self.assertEqual(p50, 0.0)
        self.assertEqual(r50, 0.0)

    def test_compute_instance_metrics_dict_matches_individual_calls(self) -> None:
        """Single IoU matrix path matches separate PR/F1 calls."""
        metrics = _reload_module("evaluation.metrics")
        gt = np.zeros((8, 8), dtype=np.int32)
        gt[1:4, 1:4] = 1
        gt[5:7, 5:7] = 2
        pr = np.zeros((8, 8), dtype=np.int32)
        pr[1:4, 1:4] = 1
        pr[5:7, 5:7] = 2
        d = metrics.compute_instance_metrics_dict(gt, pr)
        p50, r50, f50 = metrics.compute_instance_precision_recall_f1(gt, pr, 0.5)
        p75, r75, f75 = metrics.compute_instance_precision_recall_f1(gt, pr, 0.75)
        m_p, m_r, m_f = metrics.compute_instance_prf_mean_iou_sweep(gt, pr)
        self.assertAlmostEqual(d["precision_iou50"], p50)
        self.assertAlmostEqual(d["recall_iou50"], r50)
        self.assertAlmostEqual(d["f1_iou50"], f50)
        self.assertAlmostEqual(d["precision_iou75"], p75)
        self.assertAlmostEqual(d["recall_iou75"], r75)
        self.assertAlmostEqual(d["f1_iou75"], f75)
        self.assertAlmostEqual(d["mP_iou50_95"], m_p)
        self.assertAlmostEqual(d["mR_iou50_95"], m_r)
        self.assertAlmostEqual(d["mF1_iou50_95"], m_f)


class InferenceTests(unittest.TestCase):
    def test_predict_full_image_uses_training_style_edge_starts(self) -> None:
        _install_tensorflow_stub()
        inference = _reload_module("evaluation.inference")

        recorded_starts = []

        class DummyModel:
            output_shape = (None, None, None, 2)

            def predict(self, batch, verbose=0):
                recorded_starts.extend(int(value) for value in batch[:, 0, 0, 0])
                return np.zeros(
                    (batch.shape[0], batch.shape[1], batch.shape[2], 2),
                    dtype=np.float32,
                )

        image = np.arange(25, dtype=np.float32).reshape(5, 5, 1)
        inference.predict_full_image(
            model=DummyModel(),
            inputs=(image,),
            patch_size=4,
            stride=3,
            batch_size=8,
        )

        self.assertEqual(recorded_starts, [0, 1, 5, 6])

    def test_predict_full_image_matches_direct_predict_for_single_window(self) -> None:
        """One tile covering the padded image: blended path matches a single ``model.predict``."""
        _install_tensorflow_stub()
        inference = _reload_module("evaluation.inference")

        num_classes = 3

        class LogitsModel:
            output_shape = (None, None, None, num_classes)

            def predict(self, batch, verbose=0):
                _, h, w, _ = batch.shape
                yg, xg = np.mgrid[0:h, 0:w]
                cls = (yg + xg) % num_classes
                out = np.zeros(
                    (batch.shape[0], h, w, num_classes), dtype=np.float32
                )
                for b in range(batch.shape[0]):
                    for k in range(num_classes):
                        out[b, :, :, k] = (cls == k).astype(np.float32) * 10.0
                return out

        patch_size = 8
        h, w, c = 4, 4, 3
        rng = np.random.default_rng(0)
        image = rng.normal(size=(h, w, c)).astype(np.float32)
        model = LogitsModel()
        padded_h = max(h, patch_size)
        padded_w = max(w, patch_size)
        padded = np.pad(
            image,
            ((0, padded_h - h), (0, padded_w - w), (0, 0)),
            mode="constant",
        )
        direct = np.argmax(
            model.predict(padded[np.newaxis, ...], verbose=0)[0], axis=-1
        )
        direct_crop = direct[:h, :w]

        full_pred, _ = inference.predict_full_image(
            model=model,
            inputs=(image,),
            patch_size=patch_size,
            stride=patch_size,
            batch_size=1,
        )
        np.testing.assert_array_equal(full_pred, direct_crop)


def _eval_validate_args_ns(**kwargs):
    defaults = {
        "num_inputs": 1,
        "image_suffixes": ["_PPL"],
        "patch_size": 128,
        "stride": 64,
        "batch_size": 1,
        "instance_method": "cc",
        "watershed_min_distance": 1,
        "watershed_boundary_dilate_iter": 0,
        "watershed_connectivity": 1,
        "watershed_min_area_px": 0,
        "watershed_exclude_border": False,
        "watershed_ridge_level": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class EvaluateValidationTests(unittest.TestCase):
    def test_validate_args_rejects_non_positive_patch_size(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = _eval_validate_args_ns(patch_size=0)

        with self.assertRaisesRegex(ValueError, "patch_size and stride must be > 0"):
            evaluate._validate_args(args)

    def test_validate_args_rejects_invalid_num_inputs(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = _eval_validate_args_ns(
            num_inputs=3,
            image_suffixes=["_PPL", "_PPX1", "_PPX2"],
        )

        with self.assertRaisesRegex(ValueError, "num_inputs"):
            evaluate._validate_args(args)

    def test_validate_args_rejects_negative_watershed_min_area_px(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = _eval_validate_args_ns(watershed_min_area_px=-1)

        with self.assertRaisesRegex(ValueError, "watershed_min_area_px must be >= 0"):
            evaluate._validate_args(args)

    def test_validate_args_rejects_non_finite_watershed_ridge_level(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = _eval_validate_args_ns(watershed_ridge_level=float("nan"))

        with self.assertRaisesRegex(
            ValueError, "watershed_ridge_level must be finite when set"
        ):
            evaluate._validate_args(args)

    def test_validate_sample_data_rejects_mask_shape_mismatch(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        images = [np.zeros((2, 2, 3), dtype=np.float32)]
        mask = np.zeros((3, 3), dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "does not match image shape"):
            evaluate._validate_sample_data(images, mask, "mask.png")


class PlotResultsCliTests(unittest.TestCase):
    def test_main_exits_on_mismatched_quantitative_inputs(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        argv = [
            "plot_results.py",
            "--json-files",
            "a.json",
            "b.json",
            "--labels",
            "only-one",
            "--output-plot",
            "out.png",
        ]

        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                plot_results.main()

    def test_main_exits_on_incomplete_overlay_inputs(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        argv = [
            "plot_results.py",
            "--image-path",
            "image.png",
            "--gt-path",
            "gt.png",
            "--output-overlay",
            "overlay.png",
        ]

        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                plot_results.main()

    def test_main_accepts_complete_overlay_only_inputs(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        argv = [
            "plot_results.py",
            "--image-path",
            "image.png",
            "--gt-path",
            "gt.png",
            "--pred-paths",
            "pred.png",
            "--labels",
            "baseline",
            "--output-overlay",
            "overlay.png",
        ]

        with patch.object(sys, "argv", argv):
            with patch.object(plot_results, "generate_qualitative_overlay") as overlay:
                plot_results.main()

        overlay.assert_called_once_with(
            "image.png",
            "gt.png",
            ["pred.png"],
            ["baseline"],
            "overlay.png",
        )

    def test_generate_quantitative_plot_omits_error_bars_for_single_sample(
        self,
    ) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "metrics.json"
            json_path.write_text(
                json.dumps(
                    {
                        "heldout_section": {
                            "aji": 0.38,
                            "f1_iou50": 0.41,
                            "f1_iou75": 0.35,
                            "mF1_iou50_95": 0.39,
                        }
                    }
                )
            )
            output_path = Path(tmpdir) / "plot.png"

            fig = object()
            ax = SimpleNamespace(
                bar=Mock(),
                set_ylabel=Mock(),
                set_title=Mock(),
                set_xticks=Mock(),
                set_xticklabels=Mock(),
                legend=Mock(),
                grid=Mock(),
            )

            with patch.object(plot_results.plt, "subplots", return_value=(fig, ax)):
                with patch.object(plot_results.plt, "tight_layout"):
                    with patch.object(plot_results.plt, "savefig"):
                        plot_results.generate_quantitative_plot(
                            [str(json_path)], ["Baseline"], str(output_path)
                        )

            self.assertNotIn("yerr", ax.bar.call_args.kwargs)
            self.assertIn("descriptive", ax.set_title.call_args.args[0].lower())

    def test_generate_qualitative_overlay_disables_pillow_pixel_guard(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        def guarded_open(path):
            if plot_results.Image.MAX_IMAGE_PIXELS is not None:
                raise plot_results.Image.DecompressionBombError("pixel guard enabled")

            if path == "image.png":
                image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB")
            else:
                image = Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L")

            return contextlib.closing(image)

        axes = [
            SimpleNamespace(imshow=Mock(), set_title=Mock(), axis=Mock())
            for _ in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "overlay.png"

            with patch.object(plot_results.Image, "open", side_effect=guarded_open):
                plot_results.generate_qualitative_overlay(
                    "image.png",
                    "gt.png",
                    ["pred.png"],
                    ["Baseline"],
                    str(output_path),
                )

            self.assertTrue((Path(tmpdir) / "overlay_ground_truth.png").exists())
            self.assertTrue((Path(tmpdir) / "overlay_Baseline.png").exists())

    def test_blend_overlay_uses_red_tint_for_all_foreground_classes(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        image = np.zeros((1, 3, 3), dtype=np.float32)
        mask = np.array([[0, 1, 2]], dtype=np.uint8)

        overlay = plot_results.blend_overlay(image, mask)

        np.testing.assert_allclose(overlay[0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(overlay[0, 1], [0.4, 0.0, 0.0])
        np.testing.assert_allclose(overlay[0, 2], [0.4, 0.0, 0.0])

    def test_generate_qualitative_overlay_writes_ground_truth_and_one_file_per_model(
        self,
    ) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        rgb = np.full((2, 2, 3), 128, dtype=np.uint8)
        gt = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        pred_a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        pred_b = np.array([[0, 0], [2, 1]], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.png"
            gt_path = Path(tmpdir) / "gt.png"
            pred_a_path = Path(tmpdir) / "pred_a.png"
            pred_b_path = Path(tmpdir) / "pred_b.png"
            output_path = Path(tmpdir) / "overlay.png"

            Image.fromarray(rgb, mode="RGB").save(image_path)
            Image.fromarray(gt, mode="L").save(gt_path)
            Image.fromarray(pred_a, mode="L").save(pred_a_path)
            Image.fromarray(pred_b, mode="L").save(pred_b_path)
            output_path.write_text("stale montage", encoding="utf-8")

            plot_results.generate_qualitative_overlay(
                str(image_path),
                str(gt_path),
                [str(pred_a_path), str(pred_b_path)],
                ["ModelA", "ModelB"],
                str(output_path),
            )

            self.assertFalse(output_path.exists())
            self.assertTrue((Path(tmpdir) / "overlay_ground_truth.png").exists())
            self.assertTrue((Path(tmpdir) / "overlay_ModelA.png").exists())
            self.assertTrue((Path(tmpdir) / "overlay_ModelB.png").exists())

    def test_resize_overlay_arrays_downscales_large_inputs(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        rgb_img = np.zeros((5000, 2500, 3), dtype=np.float32)
        gt_mask = np.zeros((5000, 2500), dtype=np.uint8)
        pred_mask = np.zeros((5000, 2500), dtype=np.uint8)

        resized_image, resized_gt, resized_preds = plot_results._resize_overlay_arrays(
            rgb_img, gt_mask, [pred_mask], max_dim=2048
        )

        self.assertEqual(resized_image.shape[:2], resized_gt.shape)
        self.assertEqual(resized_preds[0].shape, resized_gt.shape)
        self.assertLessEqual(max(resized_image.shape[:2]), 2048)
        self.assertGreater(min(resized_image.shape[:2]), 0)


class EvaluateSampleLoadingTests(unittest.TestCase):
    def test_validate_sample_data_rejects_out_of_range_labels(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        images = [np.zeros((2, 2, 3), dtype=np.float32)]
        mask = np.array([[0, 1], [2, 3]], dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "Mask values must be in \\[0, 2\\]"):
            evaluate._validate_sample_data(images, mask, "mask.png")


class EvaluateMainTests(unittest.TestCase):
    def test_main_uses_descriptive_single_sample_output_contract(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        sample = {"id": "heldout_section", "images": ["img.png"], "mask": "mask.png"}
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        pred = np.array([[0, 1], [2, 1]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            pred_dir = Path(tmpdir) / "preds"
            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--save-predictions-dir",
                str(pred_dir),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
            ]

            stdout = io.StringIO()
            with patch.object(sys, "argv", argv):
                with patch.object(evaluate, "list_samples", return_value=[sample]):
                    with patch.object(
                        evaluate, "_load_rgb_image", return_value=np.zeros((2, 2, 3))
                    ):
                        with patch.object(
                            evaluate, "_load_raster_mask", return_value=mask
                        ):
                            with patch.object(
                                evaluate,
                                "predict_full_image",
                                return_value=(pred, np.zeros((2, 2, 3))),
                            ):
                                with contextlib.redirect_stdout(stdout):
                                    evaluate.main()

            saved = json.loads(output_json.read_text())
            self.assertIn("heldout_section", saved)
            self.assertNotIn("mean", saved)
            self.assertNotIn("AP", saved["heldout_section"])
            self.assertNotIn("AP50", saved["heldout_section"])
            self.assertEqual(saved["heldout_section"]["f1_iou50"], 1.0)
            self.assertEqual(saved["heldout_section"]["aji"], 1.0)
            self.assertTrue((pred_dir / "heldout_section_pred.png").exists())
            self.assertIn("descriptive", stdout.getvalue().lower())

    def test_main_includes_mean_metrics_for_two_samples(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        def _metric_bundle(f1: float) -> dict[str, float]:
            return {
                "precision_iou50": 1.0,
                "recall_iou50": 1.0,
                "f1_iou50": f1,
                "precision_iou75": 1.0,
                "recall_iou75": 1.0,
                "f1_iou75": f1,
                "mP_iou50_95": f1,
                "mR_iou50_95": f1,
                "mF1_iou50_95": f1,
            }

        samples = [
            {"id": "patch_a", "images": ["a.png"], "mask": "a_m.png"},
            {"id": "patch_b", "images": ["b.png"], "mask": "b_m.png"},
        ]
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        pred = np.array([[0, 1], [2, 1]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            pred_dir = Path(tmpdir) / "preds"
            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--save-predictions-dir",
                str(pred_dir),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(evaluate, "list_samples", return_value=samples):
                    with patch.object(
                        evaluate, "_load_rgb_image", return_value=np.zeros((2, 2, 3))
                    ):
                        with patch.object(
                            evaluate, "_load_raster_mask", return_value=mask
                        ):
                            with patch.object(
                                evaluate,
                                "predict_full_image",
                                return_value=(pred, np.zeros((2, 2, 3))),
                            ):
                                with patch.object(
                                    evaluate, "compute_aji", side_effect=[0.2, 0.6]
                                ):
                                    with patch.object(
                                        evaluate,
                                        "compute_instance_metrics_dict",
                                        side_effect=[
                                            _metric_bundle(0.4),
                                            _metric_bundle(0.8),
                                        ],
                                    ):
                                        evaluate.main()

            saved = json.loads(output_json.read_text())
            self.assertIn("patch_a", saved)
            self.assertIn("patch_b", saved)
            self.assertIn("mean", saved)
            self.assertAlmostEqual(saved["patch_a"]["aji"], 0.2)
            self.assertAlmostEqual(saved["patch_b"]["aji"], 0.6)
            self.assertAlmostEqual(saved["mean"]["aji"], 0.4)
            self.assertAlmostEqual(saved["mean"]["f1_iou50"], 0.6)

    def test_main_reuses_cached_prediction_skips_inference_and_model_load(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        sample = {"id": "heldout_section", "images": ["img.png"], "mask": "mask.png"}
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        pred = np.array([[0, 1], [2, 1]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            pred_dir = Path(tmpdir) / "preds"
            pred_dir.mkdir(parents=True)
            Image.fromarray(pred.astype(np.uint8)).save(
                pred_dir / "heldout_section_pred.png"
            )

            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--save-predictions-dir",
                str(pred_dir),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
            ]

            load_model = Mock(return_value=object())
            predict_full_image = Mock(
                side_effect=AssertionError("predict_full_image should not be called")
            )

            stdout = io.StringIO()
            with patch.object(sys, "argv", argv):
                with patch.object(evaluate.tf.keras.models, "load_model", load_model):
                    with patch.object(evaluate, "list_samples", return_value=[sample]):
                        with patch.object(
                            evaluate,
                            "_load_rgb_image",
                            return_value=np.zeros((2, 2, 3)),
                        ):
                            with patch.object(
                                evaluate, "_load_raster_mask", return_value=mask
                            ):
                                with patch.object(
                                    evaluate,
                                    "predict_full_image",
                                    predict_full_image,
                                ):
                                    with contextlib.redirect_stdout(stdout):
                                        evaluate.main()

            load_model.assert_not_called()
            predict_full_image.assert_not_called()
            out = stdout.getvalue()
            self.assertIn("Reusing cached prediction", out)
            self.assertIn("Instance maps", out)
            self.assertIn("AJI:", out)
            saved = json.loads(output_json.read_text())
            self.assertEqual(saved["heldout_section"]["aji"], 1.0)
            self.assertNotIn("AP", saved["heldout_section"])

    def test_main_cache_miss_calls_predict_and_loads_model_once(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        sample = {"id": "heldout_section", "images": ["img.png"], "mask": "mask.png"}
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        pred = np.array([[0, 1], [2, 1]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            pred_dir = Path(tmpdir) / "preds"
            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--save-predictions-dir",
                str(pred_dir),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
            ]

            load_model = Mock(return_value=object())
            predict_full_image = Mock(return_value=(pred, np.zeros((2, 2, 3))))

            with patch.object(sys, "argv", argv):
                with patch.object(evaluate.tf.keras.models, "load_model", load_model):
                    with patch.object(evaluate, "list_samples", return_value=[sample]):
                        with patch.object(
                            evaluate,
                            "_load_rgb_image",
                            return_value=np.zeros((2, 2, 3)),
                        ):
                            with patch.object(
                                evaluate, "_load_raster_mask", return_value=mask
                            ):
                                with patch.object(
                                    evaluate,
                                    "predict_full_image",
                                    predict_full_image,
                                ):
                                    evaluate.main()

            load_model.assert_called_once()
            predict_full_image.assert_called_once()

    def test_main_cached_prediction_raises_on_shape_mismatch(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        sample = {"id": "heldout_section", "images": ["img.png"], "mask": "mask.png"}
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        wrong_pred = np.zeros((3, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            pred_dir = Path(tmpdir) / "preds"
            pred_dir.mkdir(parents=True)
            Image.fromarray(wrong_pred).save(pred_dir / "heldout_section_pred.png")

            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--save-predictions-dir",
                str(pred_dir),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(evaluate, "list_samples", return_value=[sample]):
                    with patch.object(
                        evaluate, "_load_rgb_image", return_value=np.zeros((2, 2, 3))
                    ):
                        with patch.object(
                            evaluate, "_load_raster_mask", return_value=mask
                        ):
                            with self.assertRaisesRegex(
                                ValueError, "Cached prediction shape"
                            ):
                                evaluate.main()

    def test_main_uses_selected_instance_method_for_aji(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        sample = {"id": "heldout_section", "images": ["img.png"], "mask": "mask.png"}
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        pred = np.array([[0, 1], [2, 1]], dtype=np.int32)
        cc_instances = np.array([[0, 1], [0, 1]], dtype=np.int32)
        ws_instances = np.array([[0, 1], [0, 2]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
                "--instance-method",
                "watershed",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(evaluate, "list_samples", return_value=[sample]):
                    with patch.object(
                        evaluate, "_load_rgb_image", return_value=np.zeros((2, 2, 3))
                    ):
                        with patch.object(
                            evaluate, "_load_raster_mask", return_value=mask
                        ):
                            with patch.object(
                                evaluate,
                                "predict_full_image",
                                return_value=(pred, np.zeros((2, 2, 3))),
                            ):
                                with patch.object(
                                    evaluate,
                                    "get_instances",
                                    return_value=cc_instances,
                                ):
                                    with patch.object(
                                        evaluate,
                                        "semantic_to_instance_label_map_watershed",
                                        return_value=ws_instances,
                                    ) as ws_fn:
                                        prf_real = (
                                            evaluate.compute_instance_metrics_dict
                                        )
                                        with patch.object(
                                            evaluate,
                                            "compute_instance_metrics_dict",
                                            wraps=prf_real,
                                        ) as compute_prf:
                                            with patch.object(
                                                evaluate,
                                                "compute_aji",
                                                return_value=0.7,
                                            ) as compute_aji:
                                                evaluate.main()

            compute_aji.assert_called_once_with(cc_instances, ws_instances)
            self.assertEqual(compute_prf.call_count, 1)
            ca = compute_prf.call_args[0]
            np.testing.assert_array_equal(ca[0], cc_instances)
            np.testing.assert_array_equal(ca[1], ws_instances)
            ws_fn.assert_called_once()
            ws_kw = ws_fn.call_args[1]
            self.assertEqual(ws_kw["min_distance"], 1)
            self.assertEqual(ws_kw["boundary_dilate_iter"], 0)
            self.assertEqual(ws_kw["watershed_connectivity"], 1)
            self.assertEqual(ws_kw["min_area_px"], 0)
            self.assertFalse(ws_kw["exclude_border"])
            self.assertIsNone(ws_kw["ridge_level"])

    def test_main_passes_extended_watershed_cli_to_label_map(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        sample = {"id": "heldout_section", "images": ["img.png"], "mask": "mask.png"}
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        pred = np.array([[0, 1], [2, 1]], dtype=np.int32)
        cc_instances = np.array([[0, 1], [0, 1]], dtype=np.int32)
        ws_instances = np.array([[0, 1], [0, 2]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
                "--instance-method",
                "watershed",
                "--watershed-min-distance",
                "5",
                "--watershed-boundary-dilate-iter",
                "1",
                "--watershed-connectivity",
                "2",
                "--watershed-min-area-px",
                "10",
                "--watershed-exclude-border",
                "--watershed-ridge-level",
                "3.5",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(evaluate, "list_samples", return_value=[sample]):
                    with patch.object(
                        evaluate, "_load_rgb_image", return_value=np.zeros((2, 2, 3))
                    ):
                        with patch.object(
                            evaluate, "_load_raster_mask", return_value=mask
                        ):
                            with patch.object(
                                evaluate,
                                "predict_full_image",
                                return_value=(pred, np.zeros((2, 2, 3))),
                            ):
                                with patch.object(
                                    evaluate,
                                    "get_instances",
                                    return_value=cc_instances,
                                ):
                                    with patch.object(
                                        evaluate,
                                        "semantic_to_instance_label_map_watershed",
                                        return_value=ws_instances,
                                    ) as ws_fn:
                                        prf_real = (
                                            evaluate.compute_instance_metrics_dict
                                        )
                                        with patch.object(
                                            evaluate,
                                            "compute_instance_metrics_dict",
                                            wraps=prf_real,
                                        ) as compute_prf:
                                            with patch.object(
                                                evaluate,
                                                "compute_aji",
                                                return_value=0.7,
                                            ):
                                                evaluate.main()

            self.assertEqual(compute_prf.call_count, 1)
            ca = compute_prf.call_args[0]
            np.testing.assert_array_equal(ca[0], cc_instances)
            np.testing.assert_array_equal(ca[1], ws_instances)
            ws_fn.assert_called_once()
            ws_kw = ws_fn.call_args[1]
            self.assertEqual(ws_kw["min_distance"], 5)
            self.assertEqual(ws_kw["boundary_dilate_iter"], 1)
            self.assertEqual(ws_kw["watershed_connectivity"], 2)
            self.assertEqual(ws_kw["min_area_px"], 10)
            self.assertTrue(ws_kw["exclude_border"])
            self.assertEqual(ws_kw["ridge_level"], 3.5)


if __name__ == "__main__":
    unittest.main()
