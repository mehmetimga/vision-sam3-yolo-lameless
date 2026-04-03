"""SAM3 segmentation handler for SageMaker endpoint."""
import tempfile
import logging
from pathlib import Path

import boto3
import cv2
import numpy as np

logger = logging.getLogger("sagemaker-serve.sam3")

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    logger.warning("SAM3 not available, using bbox fallback segmentation")


class SAM3Handler:
    def __init__(self):
        self.s3 = boto3.client("s3")
        self.sam_predictor = None

        if SAM3_AVAILABLE:
            self._load_model()

    def _load_model(self):
        checkpoint_path = Path("/app/shared/models/sam3")
        if not checkpoint_path.exists() or not list(checkpoint_path.glob("*.pth")):
            logger.warning("SAM3 checkpoint not found, using fallback segmentation")
            return
        try:
            ckpt = list(checkpoint_path.glob("*.pth"))[0]
            if "vit_h" in ckpt.name:
                model_type = "vit_h"
            elif "vit_l" in ckpt.name:
                model_type = "vit_l"
            else:
                model_type = "vit_b"
            sam = sam_model_registry[model_type](checkpoint=str(ckpt))
            self.sam_predictor = SamPredictor(sam)
            logger.info(f"Loaded SAM3 model: {ckpt}")
        except Exception as e:
            logger.error(f"Failed to load SAM3: {e}")

    def handle(self, data: dict) -> dict:
        video_path = self._resolve_video(data)
        yolo_results = data.get("yolo_results", {})
        return self._segment_video(video_path, yolo_results, data.get("video_id", "unknown"))

    def _resolve_video(self, data: dict) -> Path:
        source = data.get("s3_video_path") or data.get("video_path", "")
        if source.startswith("s3://"):
            parts = source.replace("s3://", "").split("/", 1)
            suffix = Path(parts[1]).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            self.s3.download_file(parts[0], parts[1], tmp.name)
            return Path(tmp.name)
        return Path(source)

    def _segment_frame(self, image: np.ndarray, bbox: list) -> np.ndarray:
        if self.sam_predictor is not None:
            try:
                self.sam_predictor.set_image(image)
                box = np.array(bbox)
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=None, point_labels=None,
                    box=box[None, :], multimask_output=False
                )
                return masks[0]
            except Exception as e:
                logger.warning(f"SAM3 error, using fallback: {e}")

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(c) for c in bbox]
        mask[y1:y2, x1:x2] = 255
        return mask.astype(bool)

    def _extract_features(self, mask: np.ndarray) -> dict:
        mask_area = float(np.sum(mask))
        total_pixels = mask.shape[0] * mask.shape[1]
        area_ratio = mask_area / total_pixels if total_pixels > 0 else 0

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        perimeter = 0.0
        circularity = 0.0
        aspect_ratio = 0.0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest, True)
            circularity = (4 * np.pi * cv2.contourArea(largest)) / (perimeter ** 2) if perimeter > 0 else 0
            x, y, w, h = cv2.boundingRect(largest)
            aspect_ratio = w / h if h > 0 else 0

        M = cv2.moments(mask.astype(np.uint8))
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else mask.shape[1] / 2
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else mask.shape[0] / 2

        return {
            "mask_area": mask_area, "area_ratio": float(area_ratio),
            "circularity": float(circularity), "aspect_ratio": float(aspect_ratio),
            "centroid_x": float(cx), "centroid_y": float(cy),
            "perimeter": float(perimeter)
        }

    def _segment_video(self, video_path: Path, yolo_results: dict, video_id: str) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, fps // 2)
        segmentations = []
        frame_features = []
        frame_count = 0

        yolo_frame_map = {}
        for det in yolo_results.get("detections", []):
            if det.get("detections"):
                yolo_frame_map[det["frame"]] = det["detections"][0]["bbox"]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                bbox = yolo_frame_map.get(frame_count)
                if bbox:
                    mask = self._segment_frame(frame, bbox)
                    feats = self._extract_features(mask)
                    feats["frame"] = frame_count
                    feats["time"] = frame_count / fps if fps > 0 else 0
                    frame_features.append(feats)
                    segmentations.append({
                        "frame": frame_count,
                        "time": frame_count / fps if fps > 0 else 0,
                        "mask_available": True, "features": feats
                    })
                else:
                    segmentations.append({
                        "frame": frame_count,
                        "time": frame_count / fps if fps > 0 else 0,
                        "mask_available": False
                    })

            frame_count += 1

        cap.release()

        agg = {}
        if frame_features:
            agg = {
                "avg_mask_area": float(np.mean([f["mask_area"] for f in frame_features])),
                "avg_area_ratio": float(np.mean([f["area_ratio"] for f in frame_features])),
                "avg_circularity": float(np.mean([f["circularity"] for f in frame_features])),
                "avg_aspect_ratio": float(np.mean([f["aspect_ratio"] for f in frame_features])),
            }

        return {
            "video_id": video_id,
            "segmentations": segmentations,
            "aggregated_features": agg,
            "total_frames": total_frames,
            "fps": fps,
            "frames_processed": len(segmentations)
        }
