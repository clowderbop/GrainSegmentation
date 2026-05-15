import gdown
import os
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        default="data/",
        )
    parser.add_argument(
        "-u",
        "--url",
        default="https://drive.google.com/drive/folders/1yET56IAIAj616GR3cqACa1na1q2JASMF?usp=sharing",
        )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        )
    args = parser.parse_args()

    terminal_size = shutil.get_terminal_size().columns
    print("\n\n")
    print(" GrainSegmentation Dataset Downloader ".center(terminal_size, "="))
    print(f" Output location: {args.output} ".center(terminal_size))
    print(f" Force overwrite: {args.force} ".center(terminal_size))
    print("".center(terminal_size, "="))
    print("\n\n")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)

    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) == 0:
        if args.force:
            print("Deleting existing folder because --force was used")
            shutil.rmtree(out_dir)

    print("Output directory:" + out_dir)

    gdown.download_folder(
        url=args.url, use_cookies=False, output=out_dir, resume=True, remaining_ok=True
    )
