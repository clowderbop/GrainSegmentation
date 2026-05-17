import argparse
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import patches

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from common.image_io import load_tiff_channel_first
from dataset_yaml import (
    default_labels_dir as default_label_dir_for_split,
    label_map_from_yaml_names,
    load_yaml_dataset_config,
    resolve_split_dir,
)
from yolo_seg_label_io import read_yolo_seg_label_rows


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import colors

IMAGE_SUFFIXES = {".tif", ".tiff"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=4,
        )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        )
    args = parser.parse_args(argv)
    if args.num < 1:
        parser.error("--num must be at least 1")
    if args.output_dir is None:
        args.output_dir = args.dataset_dir / "visualizations"
    return args


def find_dataset_yaml(dataset_dir: Path) -> Path:
    yaml_paths = sorted(dataset_dir.glob("*.yaml")) + sorted(dataset_dir.glob("*.yml"))
    if not yaml_paths:
        raise FileNotFoundError(f"No dataset YAML found in {dataset_dir}")
    if len(yaml_paths) == 1:
        return yaml_paths[0]

    preferred = dataset_dir / f"{dataset_dir.name}.yaml"
    if preferred.exists():
        return preferred
    raise ValueError(f"Multiple dataset YAML files found in {dataset_dir}")


def load_dataset_config(dataset_dir: Path) -> tuple[Path, dict, dict[int, str]]:
    dataset_yaml = find_dataset_yaml(dataset_dir)
    dataset_root, config = load_yaml_dataset_config(dataset_yaml)
    label_map = label_map_from_yaml_names(config)
    return dataset_root, config, label_map


def collect_samples(
    dataset_root: Path, config: dict, split_name: str
) -> list[tuple[Path, Path]]:
    split_key = config.get(split_name)
    if not split_key:
        return []

    image_dir = resolve_split_dir(dataset_root, split_key)
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Missing image directory for split '{split_name}': {image_dir}"
        )

    label_dir = default_label_dir_for_split(dataset_root, split_name, image_dir)
    samples: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            samples.append((image_path, label_path))
    return samples


def load_image_preview(image_path: Path) -> tuple[np.ndarray, str | None]:
    arr = load_tiff_channel_first(image_path)
    if arr.ndim == 2:
        return arr, "gray"
    if arr.ndim == 3:
        channels = arr.shape[0]
        if channels == 1:
            return arr[0], "gray"
        if channels >= 3:
            preview = np.transpose(arr[:3], (1, 2, 0))
            return preview, None
    raise ValueError(f"Unsupported TIFF shape for visualization: {arr.shape}")


def save_visualization(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    label_map: dict[int, str],
) -> None:
    preview, cmap = load_image_preview(image_path)
    image_height, image_width = preview.shape[:2]
    polygons = read_yolo_seg_label_rows(
        label_path, image_width=image_width, image_height=image_height
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(preview, cmap=cmap)
    ax.axis("off")

    for class_id, points in polygons:
        color = tuple(channel / 255 for channel in colors(class_id, False))
        polygon = patches.Polygon(
            points, closed=True, fill=False, edgecolor=color, linewidth=2
        )
        ax.add_patch(polygon)

        anchor_x = float(np.min(points[:, 0]))
        anchor_y = float(np.min(points[:, 1]))
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        text_color = "white" if luminance < 0.5 else "black"
        ax.text(
            anchor_x,
            max(anchor_y - 4.0, 0.0),
            label_map.get(class_id, str(class_id)),
            color=text_color,
            backgroundcolor=color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, format="tiff")
    plt.close(fig)


def save_split_visualizations(
    split_name: str,
    samples: list[tuple[Path, Path]],
    output_dir: Path,
    label_map: dict[int, str],
    num_samples: int,
    rng: random.Random,
) -> int:
    if not samples:
        return 0

    selected = rng.sample(samples, k=min(num_samples, len(samples)))
    for index, (image_path, label_path) in enumerate(selected, start=1):
        output_path = output_dir / split_name / f"{index:03d}_{image_path.stem}.tif"
        save_visualization(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            label_map=label_map,
        )
    return len(selected)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    dataset_root, config, label_map = load_dataset_config(dataset_dir)
    rng = random.Random(args.seed)

    for split_name in ("train", "val"):
        samples = collect_samples(dataset_root, config, split_name)
        saved_count = save_split_visualizations(
            split_name=split_name,
            samples=samples,
            output_dir=args.output_dir.resolve(),
            label_map=label_map,
            num_samples=args.num,
            rng=rng,
        )
        print(f"Saved {saved_count} visualization(s) for {split_name}")


if __name__ == "__main__":
    main()
