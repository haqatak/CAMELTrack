import argparse
import pickle
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

def draw_bbox(frame, bbox, track_id, color):
    """Draw a bounding box and track ID on a frame."""
    x1, y1, w, h = bbox
    x2, y2 = int(x1 + w), int(y1 + h)
    x1, y1 = int(x1), int(y1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID: {track_id}"
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )
    return frame

def main():
    """
    Main function to visualize the tracking results.
    """
    parser = argparse.ArgumentParser(
        description="Visualize tracking results on a video."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the tracking results file (.pklz).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output video. Defaults to a new file in the same directory as the input video.",
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.is_file():
        print(f"Error: Video file not found at '{video_path}'", file=sys.stderr)
        sys.exit(1)

    results_path = Path(args.results_path)
    if not results_path.is_file():
        print(f"Error: Results file not found at '{results_path}'", file=sys.stderr)
        sys.exit(1)

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = video_path.parent / f"{video_path.stem}_tracked.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the tracking results
    with open(results_path, "rb") as f:
        tracker_state = pickle.load(f)

    detections = tracker_state.detections_pred

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'", file=sys.stderr)
        sys.exit(1)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create video writer
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Create a color map for the track IDs
    colors = {}

    frame_idx = 0
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_detections = detections[detections.frame_id == frame_idx]

            for _, row in frame_detections.iterrows():
                track_id = row.get("track_id")
                if pd.notna(track_id):
                    track_id = int(track_id)
                    if track_id not in colors:
                        colors[track_id] = tuple(map(int, (hash(str(track_id)) % 255, hash(str(track_id*2)) % 255, hash(str(track_id*3)) % 255)))

                    bbox = row.get("bbox_ltwh")
                    if bbox is not None:
                        frame = draw_bbox(frame, bbox, track_id, colors[track_id])

            out.write(frame)
            pbar.update(1)
            frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nVisualization complete! Video saved to: {output_path}")

if __name__ == "__main__":
    try:
        import pandas as pd
        import cv2
        from tqdm import tqdm
    except ImportError:
        print("Error: 'pandas', 'opencv-python', or 'tqdm' is not installed.", file=sys.stderr)
        print("Please install them with: pip install pandas opencv-python tqdm", file=sys.stderr)
        sys.exit(1)
    main()