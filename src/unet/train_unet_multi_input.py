import argparse
from pathlib import Path

from common.arg_errors import raise_cli_argument_error
from common.manifest_io import collect_manifest_unet_samples, load_dataset_manifest
from unet.train import train_model


def _print_start_message(
    args: argparse.Namespace, *, stride: int, split_tile_size: int
) -> None:
    fields = list(vars(args).items())
    fields.extend(
        [
            ("stride", stride),
            ("effective_split_tile_size", split_tile_size),
            ("mixed_precision_enabled", not args.no_mixed_precision),
        ]
    )
    key_width = max(len(key) for key, _ in fields)
    border = "=" * 80

    print(border)
    print("Training Pipeline Start")
    print(border)
    for key, value in fields:
        print(f"{key:<{key_width}} : {value}")
    print(border)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--patch-overlap", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--tune-epochs", type=int, default=30)
    parser.add_argument("--run-name", default="default_run")
    parser.add_argument("--tuning-dir", default="tuning_dir")
    parser.add_argument("--mask-ext", default=None)
    parser.add_argument("--mask-stem-suffix", default="")
    parser.add_argument("--num-inputs", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-tile-size", type=int, default=4096)
    parser.add_argument("--split-coverage-bins", type=int, default=8)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    return parser.parse_args(argv)


def _resolve_training_samples(
    args: argparse.Namespace,
) -> tuple[list[dict], int]:
    doc = load_dataset_manifest(args.manifest)
    row = doc.samples[0]
    if row.images is None:
        raise_cli_argument_error("Training manifest requires images rows")
    num_inputs = len(row.images)
    samples = collect_manifest_unet_samples(
        doc,
        mask_dir=args.mask_dir,
        mask_ext=args.mask_ext,
        mask_stem_suffix=args.mask_stem_suffix,
    )
    return samples, num_inputs


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    samples, num_inputs = _resolve_training_samples(args)

    if args.num_inputs is not None and args.num_inputs != num_inputs:
        raise ValueError(
            f"--num-inputs {args.num_inputs} does not match resolved {num_inputs}"
        )
    args.num_inputs = num_inputs

    if args.num_inputs not in {1, 2, 7}:
        raise ValueError("--num-inputs must be one of: 1, 2, 7.")
    if args.checkpoint and args.resume:
        raise ValueError("Use only one of --checkpoint or --resume.")
    if args.split_tile_size < 0:
        raise ValueError("--split-tile-size must be >= 0")
    if args.patch_overlap < 0 or args.patch_overlap >= 1:
        raise ValueError("--patch-overlap must be in [0, 1).")
    if args.validation_fraction <= 0 or args.validation_fraction >= 1:
        raise ValueError("--validation-fraction must be in (0, 1).")

    split_tile_size = args.split_tile_size or args.patch_size * 2
    stride = int(args.patch_size * (1 - args.patch_overlap))
    _print_start_message(args, stride=stride, split_tile_size=split_tile_size)

    train_model(
        mask_dir="",
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        output_model_path=args.output_model,
        patch_size=args.patch_size,
        stride=stride,
        tune_epochs=args.tune_epochs,
        final_epochs=args.epochs,
        image_suffixes=[],
        mask_ext=args.mask_ext,
        mask_stem_suffix=args.mask_stem_suffix,
        split_tile_size=split_tile_size,
        split_coverage_bins=args.split_coverage_bins,
        num_inputs=num_inputs,
        run_name=args.run_name,
        tuning_dir=args.tuning_dir,
        validation_fraction=args.validation_fraction,
        random_state=args.seed,
        use_mixed_precision=not args.no_mixed_precision,
        max_trials=args.max_trials,
        skip_tuning=args.skip_tuning,
        samples=samples,
    )


if __name__ == "__main__":
    main()
