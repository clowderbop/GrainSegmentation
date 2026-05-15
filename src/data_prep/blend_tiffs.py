import argparse
import os
import sys
import glob
import numpy as np
import tifffile
from tqdm import tqdm


def get_max_value(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        return 1.0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def process_chunk(base_chunk, other_chunks, dtype):
    max_val = get_max_value(dtype)


    blended = 1.0 - (base_chunk.astype(np.float32) / max_val)


    for chunk in other_chunks:
        normalized_chunk = chunk.astype(np.float32) / max_val
        blended *= 1.0 - normalized_chunk

    blended = 1.0 - blended


    if np.issubdtype(dtype, np.integer):

        blended = np.clip(blended * max_val, 0, max_val).astype(dtype)
    else:
        blended = np.clip(blended, 0.0, 1.0).astype(dtype)

    return blended


def blend_tiffs(
    input_dir, output_file, base_file=None, chunk_size=2048, exclude_files=None
):
    if exclude_files is None:
        exclude_files = []

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)


    search_patterns = [
        os.path.join(input_dir, "*.tif"),
        os.path.join(input_dir, "*.tiff"),
    ]
    tiff_files = []
    for pattern in search_patterns:
        tiff_files.extend(glob.glob(pattern))


    output_abs = os.path.abspath(output_file)
    tiff_files = [f for f in tiff_files if os.path.abspath(f) != output_abs]


    exclude_abs = [os.path.abspath(f) for f in exclude_files]
    tiff_files = [f for f in tiff_files if os.path.abspath(f) not in exclude_abs]

    if not tiff_files:
        print(f"Error: No TIFF files found in '{input_dir}'.")
        sys.exit(1)

    tiff_files.sort()


    if base_file:
        if base_file not in tiff_files:
            if os.path.abspath(base_file) in [os.path.abspath(f) for f in tiff_files]:

                base_idx = [os.path.abspath(f) for f in tiff_files].index(
                    os.path.abspath(base_file)
                )
                base_file = tiff_files[base_idx]
            else:
                if not os.path.isfile(base_file):
                    print(f"Error: Base file '{base_file}' does not exist.")
                    sys.exit(1)
                tiff_files.insert(0, base_file)
    else:
        base_file = tiff_files[0]

    print(f"Found {len(tiff_files)} images to blend.")
    print(f"Base image: {base_file}")

    other_files = [f for f in tiff_files if f != base_file]

    if not other_files:
        print("Warning: Only one image found. Nothing to blend.")

        sys.exit(0)


    try:
        with tifffile.TiffFile(base_file) as tif:
            base_page = tif.pages[0]
            shape = base_page.shape
            dtype = base_page.dtype

            print(f"Image properties - Shape: {shape}, Dtype: {dtype}")
    except Exception as e:
        print(f"Error reading base file metadata: {e}")
        sys.exit(1)


    print("Validating input images...")
    for f in other_files:
        try:
            with tifffile.TiffFile(f) as tif:
                page = tif.pages[0]
                if page.shape != shape:
                    print(
                        f"Error: Dimension mismatch in '{f}'. Expected {shape}, got {page.shape}."
                    )
                    sys.exit(1)
                if page.dtype != dtype:
                    print(
                        f"Error: Data type mismatch in '{f}'. Expected {dtype}, got {page.dtype}."
                    )
                    sys.exit(1)
        except Exception as e:
            print(f"Error reading '{f}': {e}")
            sys.exit(1)

    print("Validation successful.")


    base_mmap = tifffile.memmap(base_file, mode="r")
    other_mmaps = [tifffile.memmap(f, mode="r") for f in other_files]


    print(f"Creating output file: {output_file}")


    out_mmap = tifffile.memmap(
        output_file,
        shape=shape,
        dtype=dtype,
        photometric="rgb" if len(shape) == 3 and shape[-1] in (3, 4) else "minisblack",
    )


    if len(shape) == 2:
        height, width = shape
        channels = 1
    elif len(shape) == 3:

        height, width, channels = shape
    else:
        print(f"Error: Unsupported image shape {shape}. Expected 2D or 3D (HWC).")
        sys.exit(1)

    y_steps = range(0, height, chunk_size)
    x_steps = range(0, width, chunk_size)
    total_chunks = len(y_steps) * len(x_steps)

    print(f"Processing in {total_chunks} chunks of size {chunk_size}x{chunk_size}...")

    with tqdm(total=total_chunks, desc="Blending") as pbar:
        for y in y_steps:
            y_end = min(y + chunk_size, height)
            for x in x_steps:
                x_end = min(x + chunk_size, width)


                if len(shape) == 2:
                    b_chunk = base_mmap[y:y_end, x:x_end]
                    o_chunks = [m[y:y_end, x:x_end] for m in other_mmaps]
                else:
                    b_chunk = base_mmap[y:y_end, x:x_end, :]
                    o_chunks = [m[y:y_end, x:x_end, :] for m in other_mmaps]


                blended_chunk = process_chunk(b_chunk, o_chunks, dtype)

                if len(shape) == 2:
                    out_mmap[y:y_end, x:x_end] = blended_chunk
                else:
                    out_mmap[y:y_end, x:x_end, :] = blended_chunk

                pbar.update(1)


    out_mmap.flush()
    print("Blending complete!")


def main():
    parser = argparse.ArgumentParser(
        )
    parser.add_argument("input_dir", )
    parser.add_argument("output_file", )
    parser.add_argument(
        "--base",
        default=None,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        )

    args = parser.parse_args()

    blend_tiffs(
        args.input_dir, args.output_file, args.base, args.chunk_size, args.exclude
    )


if __name__ == "__main__":
    main()
