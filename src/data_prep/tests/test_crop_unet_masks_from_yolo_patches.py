import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "crop_unet_masks_from_yolo_patches.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("crop_unet_masks", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CropUnetMasksFromYoloPatchesTests(unittest.TestCase):
    def test_parse_patch_stem(self) -> None:
        m = _load_module()
        self.assertEqual(m.parse_patch_stem("region_0001_y00100_x00200"), (1, 100, 200))

    def test_parse_patch_stem_rejects_bad_name(self) -> None:
        m = _load_module()
        with self.assertRaises(ValueError):
            m.parse_patch_stem("foobar")

    def test_end_to_end_single_patch(self) -> None:
        m = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h, w, pz, tz = 10, 10, 10, 10
            ref = root / "ref.tif"
            arr = np.zeros((1, h, w), dtype=np.uint8)
            tifffile.imwrite(ref, arr, metadata={"axes": "CYX"})

            mask = np.arange(h * w, dtype=np.uint8).reshape(h, w)
            mask_path = root / "mask.tif"
            tifffile.imwrite(mask_path, mask, compression="deflate")

            yolo_dir = root / "yolo"
            yolo_dir.mkdir()
            patch_img = np.full((1, pz, pz), 99, dtype=np.uint8)
            stem = "region_0000_y00000_x00000"
            tifffile.imwrite(
                yolo_dir / f"{stem}.tif", patch_img, metadata={"axes": "CYX"}
            )

            out_img = root / "out_i"
            out_msk = root / "out_m"
            m.main(
                [
                    "--reference-tiff",
                    str(ref),
                    "--reference-mask",
                    str(mask_path),
                    "--yolo-images-dir",
                    str(yolo_dir),
                    "--output-images-dir",
                    str(out_img),
                    "--output-masks-dir",
                    str(out_msk),
                    "--patch-size",
                    str(pz),
                    "--tile-size",
                    str(tz),
                    "--image-suffix",
                    "_PPL",
                ]
            )

            written_mask = tifffile.imread(out_msk / f"{stem}_labels.tif")
            np.testing.assert_array_equal(written_mask, mask)

            self.assertTrue((out_img / f"{stem}_PPL.tif").is_file())


if __name__ == "__main__":
    unittest.main()
