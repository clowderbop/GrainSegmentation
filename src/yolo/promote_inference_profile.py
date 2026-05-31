"""Promote grid profile-selection winner into configs/test_inference.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.test_inference import inference_recipe_path
from yolo.inference_profile_tune import load_grid_winner, promote_profile_to_recipe


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--winner-json",
        type=Path,
        required=True,
        help="grid/winner.json from profile tune run.",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=None,
        help="Target test inference recipe (default: configs/test_inference.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    recipe_path = args.recipe or inference_recipe_path()
    winner_path = args.winner_json
    candidate = load_grid_winner(winner_path)
    promote_profile_to_recipe(candidate, recipe_path)
    print(f"Promoted YOLO inference profile into {recipe_path}")


if __name__ == "__main__":
    main()
