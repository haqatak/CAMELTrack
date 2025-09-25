import argparse
import subprocess
import sys
from pathlib import Path

def main():
    """
    Main function to run the tracking script.
    """
    parser = argparse.ArgumentParser(
        description="Track players in a video using CAMELTrack."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file (e.g., soccer_match.mp4).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="camel_bbox_app_kps_sportsmot.ckpt",
        help="Name of the model checkpoint file to use.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/tracking_results",
        help="Directory to save the tracking results.",
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.is_file():
        print(f"Error: Video file not found at '{video_path}'")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_tracking.pklz"

    model_path = f"pretrained_models/camel/{args.model_name}"

    print("Starting video tracking...")
    print(f"  Video: {video_path}")
    print(f"  Model: {args.model_name}")
    print(f"  Output will be saved to: {output_path}")

    # Construct the command to run tracklab with Hydra overrides
    command = [
        "uv", "run", "tracklab",
        "-cn", "cameltrack",
        "dataset=soccer_match",
        f"dataset.video_path={str(video_path.resolve())}",
        f"modules.track.checkpoint_path={model_path}",
        f"state.save_file={str(output_path.resolve())}",
    ]

    try:
        subprocess.run(command, check=True)
        print("\nTracking complete!")
        print(f"Results saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred while running the tracking process: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: 'uv' or 'tracklab' command not found.", file=sys.stderr)
        print("Please make sure you are in the correct environment and have the necessary packages installed.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()