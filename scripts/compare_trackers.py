import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.compare import compare_dataset_trackers, compare_upload_trackers


def main():
    parser = argparse.ArgumentParser(
        description="Compare the original DeepSORT tracker with the custom DeepSORT tracker."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str, help="Dataset name, for example: VNTraffic or AICC22-Custom")
    group.add_argument("--video", type=str, help="Path to an uploaded or custom video")

    parser.add_argument("--name", type=str, default=None, help="Run name when using --video")
    parser.add_argument("--model", type=str, default="models/yolov5n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-frames", type=int, default=None)

    args = parser.parse_args()

    if args.dataset:
        summary = compare_dataset_trackers(
            dataset_name=args.dataset,
            yolo_model_path=args.model,
            conf_threshold=args.conf,
            image_size=args.imgsz,
            device=args.device,
            max_frames=args.max_frames,
        )
    else:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video not found: {args.video}")

        run_name = args.name or Path(args.video).stem

        summary = compare_upload_trackers(
            input_video_path=args.video,
            run_name=run_name,
            yolo_model_path=args.model,
            conf_threshold=args.conf,
            image_size=args.imgsz,
            device=args.device,
            max_frames=args.max_frames,
        )

    print(json.dumps(summary["comparison"], ensure_ascii=False, indent=4))
    print()
    print(f"Summary saved to: {summary['summary_path']}")


if __name__ == "__main__":
    main()
