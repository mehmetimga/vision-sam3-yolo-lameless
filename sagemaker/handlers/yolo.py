"""YOLO inference handler for SageMaker endpoint."""
import tempfile
import logging
from pathlib import Path

import boto3
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("sagemaker-serve.yolo")


class YOLOHandler:
    def __init__(self):
        self.s3 = boto3.client("s3")

        model_path = Path("/app/shared/models/yolo")
        if model_path.exists() and list(model_path.glob("*.pt")):
            model_file = list(model_path.glob("*.pt"))[0]
            self.model = YOLO(str(model_file))
            logger.info(f"Loaded custom YOLO model: {model_file}")
        else:
            self.model = YOLO("yolov8n.pt")
            logger.info("Using pretrained YOLOv8n model")

        self.confidence_threshold = 0.5

    def handle(self, data: dict) -> dict:
        video_path = self._resolve_video(data)
        results = self._detect_in_video(video_path)
        results["video_id"] = data.get("video_id", "unknown")
        return results

    def _resolve_video(self, data: dict) -> Path:
        source = data.get("s3_video_path") or data.get("video_path", "")
        if source.startswith("s3://"):
            return self._download_s3(source)
        return Path(source)

    def _download_s3(self, s3_uri: str) -> Path:
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        suffix = Path(key).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        self.s3.download_file(bucket, key, tmp.name)
        return Path(tmp.name)

    def _detect_in_video(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, fps // 2)
        detections = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                results = self.model(frame, verbose=False, conf=self.confidence_threshold)
                frame_dets = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names.get(cls, f"class_{cls}")
                        frame_dets.append({
                            "frame": frame_count,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "class": class_name,
                            "class_id": cls
                        })

                if frame_dets:
                    detections.append({
                        "frame": frame_count,
                        "time": frame_count / fps if fps > 0 else 0,
                        "detections": frame_dets
                    })

            frame_count += 1

        cap.release()
        features = self._compute_features(detections, total_frames, fps)

        return {
            "detections": detections,
            "features": features,
            "total_frames": total_frames,
            "fps": fps,
            "frames_processed": len(detections)
        }

    def _compute_features(self, detections: list, total_frames: int, fps: float) -> dict:
        if not detections:
            return {}

        all_boxes, confidences = [], []
        for frame_data in detections:
            for det in frame_data["detections"]:
                all_boxes.append(det["bbox"])
                confidences.append(det["confidence"])

        if not all_boxes:
            return {}

        boxes = np.array(all_boxes)
        confs = np.array(confidences)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        stability = 1.0 / (1.0 + np.std(cx) + np.std(cy))

        return {
            "num_detections": len(boxes),
            "avg_confidence": float(np.mean(confs)),
            "max_confidence": float(np.max(confs)),
            "min_confidence": float(np.min(confs)),
            "avg_box_area": float(np.mean(areas)),
            "avg_box_width": float(np.mean(widths)),
            "avg_box_height": float(np.mean(heights)),
            "position_stability": float(stability),
            "avg_center_x": float(np.mean(cx)),
            "avg_center_y": float(np.mean(cy)),
            "detection_rate": len(detections) / total_frames if total_frames > 0 else 0
        }
