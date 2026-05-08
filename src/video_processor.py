import os
import csv
import time
from typing import Dict, Any

import cv2

from src.detector import YOLODetector
from src.tracker import DeepSORTTracker
from src.utils import convert_video_to_h264


class VideoProcessor:
    """
    VideoProcessor xử lý traffic video upload cho bài toán vehicle tracking.

    Pipeline:
        Input traffic video
        → YOLODetector phát hiện phương tiện
        → DeepSORTTracker gán vehicle ID
        → Lưu output video đã overlay bounding box + ID + class + confidence
        → Convert output video sang H.264 để web phát được
        → Lưu vehicle tracking result txt

    File tracking txt có format:
        frame,id,x,y,w,h,conf,class,visibility

    Trong đó:
        class = 2  -> car
        class = 3  -> motorcycle
        class = 5  -> bus
        class = 7  -> truck
        class = -1 -> predicted, track được DeepSORT dự đoán tạm thời
    """

    def __init__(
        self,
        yolo_model_path: str = "models/yolov5n.pt",
        conf_threshold: float = 0.35,
        image_size: int = 416,
        device: str = "cpu"
    ):
        self.detector = YOLODetector(
            model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            image_size=image_size,
            device=device
        )

        self.tracker = DeepSORTTracker(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.4
        )

    def process(
        self,
        input_video_path: str,
        output_video_path: str,
        output_txt_path: str
    ) -> Dict[str, Any]:
        """
        Xử lý một traffic video.

        Args:
            input_video_path:
                Đường dẫn video input.

            output_video_path:
                Đường dẫn video output sau khi vẽ bounding box + vehicle ID.

            output_txt_path:
                Đường dẫn file txt lưu kết quả vehicle tracking.

        Returns:
            Dict chứa thông tin xử lý:
                total_frames
                elapsed_time
                process_fps
                input_video_path
                output_video_path
                output_txt_path
                width
                height
                original_fps
        """

        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Không tìm thấy video: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {input_video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            original_fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

        # OpenCV ghi file tạm bằng mp4v trước.
        # Sau khi xử lý xong, ta convert sang H.264 để trình duyệt phát được.
        base_name, ext = os.path.splitext(output_video_path)
        temp_output_video_path = f"{base_name}_temp{ext}"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_video = cv2.VideoWriter(
            temp_output_video_path,
            fourcc,
            original_fps,
            (width, height)
        )

        if not writer_video.isOpened():
            cap.release()
            raise RuntimeError(f"Không tạo được video output tạm: {temp_output_video_path}")

        frame_id = 0
        start_time = time.time()

        try:
            with open(output_txt_path, mode="w", newline="") as txt_file:
                csv_writer = csv.writer(txt_file)

                # Header nội bộ để web/statistics đọc dễ hơn.
                # Nếu cần nộp theo MOTChallenge chuẩn, có thể export thêm bản không header.
                csv_writer.writerow([
                    "frame",
                    "id",
                    "x",
                    "y",
                    "w",
                    "h",
                    "conf",
                    "class",
                    "visibility"
                ])

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_id += 1

                    detections = self.detector.detect(frame)
                    tracks = self.tracker.update(detections, frame)
                    output_frame = self.tracker.draw_tracks(frame, tracks)

                    for track in tracks:
                        track_id = track["track_id"]
                        x, y, w, h = track["bbox_xywh"]

                        confidence = track.get("confidence", 0.0)
                        class_id = track.get("class_id", -1)
                        visibility = track.get("visibility", 0)

                        # Format gần MOTChallenge:
                        # frame,id,x,y,w,h,conf,class,visibility
                        csv_writer.writerow([
                            frame_id,
                            track_id,
                            x,
                            y,
                            w,
                            h,
                            round(float(confidence), 4),
                            int(class_id),
                            int(visibility)
                        ])

                    writer_video.write(output_frame)

                    if frame_id % 20 == 0:
                        print(f"Đã xử lý {frame_id}/{total_input_frames} frames")

        finally:
            cap.release()
            writer_video.release()

        end_time = time.time()
        elapsed_time = end_time - start_time
        process_fps = frame_id / elapsed_time if elapsed_time > 0 else 0

        # Convert video tạm sang H.264.
        convert_video_to_h264(
            input_path=temp_output_video_path,
            output_path=output_video_path
        )

        # Xóa file tạm sau khi convert xong.
        if os.path.exists(temp_output_video_path):
            os.remove(temp_output_video_path)

        return {
            "total_frames": frame_id,
            "elapsed_time": elapsed_time,
            "process_fps": process_fps,
            "input_video_path": input_video_path,
            "output_video_path": output_video_path,
            "output_txt_path": output_txt_path,
            "width": width,
            "height": height,
            "original_fps": original_fps
        }