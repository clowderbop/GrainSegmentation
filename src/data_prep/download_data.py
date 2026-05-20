import argparse
import os
import shutil
import subprocess
import sys

import gdown

DEFAULT_URL = (
    "https://drive.google.com/uc?id=1Q3VLp5j2N34RYv2Dcbbb1B8QZluZetef"
)
DEFAULT_ARCHIVE_NAME = "uncropped.tar.lz4"


def _normalize_extracted_tree(output_dir: str) -> None:
    """If the archive used a single wrapper directory, hoist files up."""
    entries = os.listdir(output_dir)
    if len(entries) != 1:
        return
    wrapper = os.path.join(output_dir, entries[0])
    if not os.path.isdir(wrapper) or entries[0] not in ("uncropped", "dataset"):
        return
    for name in os.listdir(wrapper):
        shutil.move(os.path.join(wrapper, name), os.path.join(output_dir, name))
    os.rmdir(wrapper)


def extract_tar_lz4(archive_path: str, output_dir: str) -> None:
    lz4 = shutil.which("lz4")
    if lz4 is None:
        raise RuntimeError(
            "lz4 not found on PATH. Load an lz4 module before running, "
            "e.g. `module load lz4/1.9.4-GCCcore-12.3.0`."
        )
    tar = shutil.which("tar")
    if tar is None:
        raise RuntimeError("tar not found on PATH")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracting {archive_path} -> {output_dir}")

    lz4_proc = subprocess.Popen(
        [lz4, "-d", "-c", archive_path],
        stdout=subprocess.PIPE,
    )
    try:
        subprocess.run(
            [tar, "-xf", "-", "-C", output_dir],
            stdin=lz4_proc.stdout,
            check=True,
        )
    finally:
        if lz4_proc.stdout is not None:
            lz4_proc.stdout.close()
        lz4_returncode = lz4_proc.wait()
    if lz4_returncode != 0:
        raise subprocess.CalledProcessError(lz4_returncode, lz4_proc.args)

    _normalize_extracted_tree(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download uncropped.tar.lz4 from Google Drive and extract "
            "into the dataset uncropped directory."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/",
        help="Directory to extract uncropped dataset files into.",
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help=(
            "Directory for the downloaded archive. "
            "Defaults to the parent of --output (e.g. dataset/)."
        ),
    )
    parser.add_argument(
        "-u",
        "--url",
        default=DEFAULT_URL,
        help="Google Drive file URL or share link.",
    )
    parser.add_argument(
        "--archive-name",
        default=DEFAULT_ARCHIVE_NAME,
        help="Filename to save the downloaded archive as.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-download and re-extract even if outputs already exist.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download only; do not extract the archive.",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    archive_dir = (
        os.path.abspath(args.archive_dir)
        if args.archive_dir
        else os.path.dirname(output_dir)
    )
    archive_path = os.path.join(archive_dir, args.archive_name)

    terminal_size = shutil.get_terminal_size().columns
    print("\n\n")
    print(" GrainSegmentation Dataset Downloader ".center(terminal_size, "="))
    print(f" Extract to: {output_dir} ".center(terminal_size))
    print(f" Archive: {archive_path} ".center(terminal_size))
    print(f" Force overwrite: {args.force} ".center(terminal_size))
    print("".center(terminal_size, "="))
    print("\n\n")

    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if args.force:
            print(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(
                f"Output directory is not empty: {output_dir}. "
                "Use --force to overwrite."
            )
            sys.exit(1)

    os.makedirs(archive_dir, exist_ok=True)
    if os.path.isfile(archive_path):
        if args.force:
            print(f"Removing existing archive: {archive_path}")
            os.remove(archive_path)
        else:
            print(f"Using existing archive: {archive_path}")
    else:
        print(f"Downloading {args.archive_name} from Google Drive...")
        gdown.download(url=args.url, output=archive_path, fuzzy=True, resume=True)

    if args.skip_extract:
        print("Skipping extraction (--skip-extract).")
        sys.exit(0)

    extract_tar_lz4(archive_path, output_dir)
    print(f"Done. Uncropped dataset is at {output_dir}")
