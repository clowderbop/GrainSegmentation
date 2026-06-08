"""Phase 0: manifest contract dataclass guards."""

from pathlib import Path

from common.manifest_io import DatasetManifest, ManifestSampleRow


def test_load_dataset_manifest_round_trip(tmp_path: Path) -> None:
    """INTENT: write_dataset_manifest and load_dataset_manifest preserve multi-image sample rows."""
    from common.manifest_io import load_dataset_manifest, write_dataset_manifest

    source = DatasetManifest(
        schema_version=1,
        variant="PPL+PPXblend",
        unit="whole",
        grainseg_root=str(tmp_path),
        path_base="grainseg_root",
        samples=(
            ManifestSampleRow(
                sample_id="test",
                images=(
                    "dataset/test/test_PPL.tif",
                    "dataset/test/test_PPXblend.tif",
                ),
                gt_gpkg="dataset/test/test_labels.gpkg",
                gt_origin="whole_image",
            ),
        ),
    )
    path = tmp_path / "m.json"
    write_dataset_manifest(path, source)
    loaded = load_dataset_manifest(path)
    assert loaded.variant == "PPL+PPXblend"
    assert loaded.samples[0].images is not None
    assert len(loaded.samples[0].images) == 2
