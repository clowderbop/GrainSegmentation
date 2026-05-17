import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from common.image_io import TIFF_SUFFIXES, to_channel_first_uint8


VALID_SUFFIXES = TIFF_SUFFIXES


def _to_channel_first_uint8(image: np.ndarray) -> np.ndarray:
    return to_channel_first_uint8(
        image, channel_multiple=3, description="a stacked TIFF"
    )


def split_tiff_channels(
    input_file: str | Path,
    output_dir: str | Path,
    prefix: str | None = None,
) -> list[Path]:
    input_path = Path(input_file)
    output_path = Path(output_dir)

    if not input_path.is_file():
        raise ValueError(f"Input TIFF does not exist: {input_path}")

    if input_path.suffix.lower() not in VALID_SUFFIXES:
        raise ValueError("Input file must end with .tif or .tiff")

    stacked = tifffile.imread(input_path)
    channel_first = _to_channel_first_uint8(stacked)

    if channel_first.shape[0] == 0 or channel_first.shape[0] % 3 != 0:
        raise ValueError(
            "Stacked TIFF must contain a non-empty number of channels divisible by 3."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    output_prefix = prefix or input_path.stem

    written_files = []
    for index, start in enumerate(range(0, channel_first.shape[0], 3)):
        rgb = np.transpose(channel_first[start : start + 3], (1, 2, 0))
        image_path = output_path / f"{output_prefix}_{index:03d}.tif"
        tifffile.imwrite(image_path, rgb, photometric="rgb")
        written_files.append(image_path)

    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(
        )
    parser.add_argument("input_file", )
    parser.add_argument("output_dir", )
    parser.add_argument(
        "--prefix",
        default=None,
        )
    args = parser.parse_args()

    try:
        output_files = split_tiff_channels(
            args.input_file,
            args.output_dir,
            prefix=args.prefix,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved {len(output_files)} RGB TIFF files to {args.output_dir}")


if __name__ == "__main__":
    main()
