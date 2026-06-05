"""Sparse build_instance_iou_matrix parity vs dense reference and bundle stability."""

from __future__ import annotations

import numpy as np
import pytest

from common.instance_metric_bundle import (
    INSTANCE_METRIC_BUNDLE_KEYS,
    compute_instance_metric_bundle,
)
from common.metrics import build_instance_iou_matrix
from common.tests.dense_iou_reference import (
    compute_instance_metric_bundle_dense_reference,
    dense_build_instance_iou_matrix,
)
from common.tests.merged_view_fixtures import (
    BUNDLE_FIXTURE_BUILDERS,
    blank_map,
    get_bundle_fixture,
    paint_box,
)


def _assert_matrix_matches_dense_reference(
    gt: np.ndarray, pred: np.ndarray
) -> None:
    sparse_mat, sparse_gt_ids, sparse_pred_ids = build_instance_iou_matrix(gt, pred)
    ref_mat, ref_gt_ids, ref_pred_ids = dense_build_instance_iou_matrix(gt, pred)
    assert sparse_gt_ids == ref_gt_ids
    assert sparse_pred_ids == ref_pred_ids
    assert sparse_mat.shape == ref_mat.shape
    np.testing.assert_allclose(sparse_mat, ref_mat)


def _assert_bundle_matches_dense_reference(gt: np.ndarray, pred: np.ndarray) -> None:
    bundle = compute_instance_metric_bundle(gt, pred)
    reference = compute_instance_metric_bundle_dense_reference(gt, pred)
    assert tuple(bundle.keys()) == INSTANCE_METRIC_BUNDLE_KEYS
    for key in INSTANCE_METRIC_BUNDLE_KEYS:
        assert bundle[key] == pytest.approx(reference[key]), key


def test_sparse_iou_matrix_empty_maps() -> None:
    _assert_matrix_matches_dense_reference(blank_map(8, 8), blank_map(8, 8))


def test_sparse_iou_matrix_single_perfect_match() -> None:
    gt = blank_map(16, 16)
    pred = blank_map(16, 16)
    paint_box(gt, 1, 2, 2, 10, 10)
    paint_box(pred, 1, 2, 2, 10, 10)
    _assert_matrix_matches_dense_reference(gt, pred)


def test_sparse_iou_matrix_non_overlapping_instances_are_zero() -> None:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 4, 4, 14, 14)
    paint_box(gt, 2, 18, 18, 28, 28)
    paint_box(pred, 1, 4, 4, 14, 14)

    mat, gt_ids, pred_ids = build_instance_iou_matrix(gt, pred)
    assert gt_ids == [1, 2]
    assert pred_ids == [1]
    assert mat[0, 0] == pytest.approx(1.0)
    assert mat[1, 0] == pytest.approx(0.0)


def test_sparse_iou_matrix_split_merge_overlap() -> None:
    gt, pred = get_bundle_fixture("split_merge")
    _assert_matrix_matches_dense_reference(gt, pred)


def test_sparse_iou_matrix_gapped_label_ids() -> None:
    gt = blank_map(24, 24)
    pred = blank_map(24, 24)
    paint_box(gt, 7, 2, 2, 10, 10)
    paint_box(pred, 42, 2, 2, 10, 10)
    _assert_matrix_matches_dense_reference(gt, pred)


@pytest.mark.parametrize("fixture_name", list(BUNDLE_FIXTURE_BUILDERS))
def test_compute_instance_metric_bundle_unchanged_after_sparse_matrix(
    fixture_name: str,
) -> None:
    gt, pred = get_bundle_fixture(fixture_name)
    _assert_matrix_matches_dense_reference(gt, pred)
    _assert_bundle_matches_dense_reference(gt, pred)
