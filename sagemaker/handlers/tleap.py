"""T-LEAP pose estimation handler for SageMaker endpoint."""
import tempfile
import logging
from pathlib import Path

import boto3
import cv2
import numpy as np

logger = logging.getLogger("sagemaker-serve.tleap")

try:
    from ultralytics import YOLO as YOLOPose
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics not available for pose estimation")

KEYPOINT_NAMES = [
    'left_ear_base', 'neck', 'withers', 'mid_back',
    'right_hind_hip', 'right_hind_mid_leg', 'right_hind_fetlock',
    'right_hind_hoof', 'left_hind_hip', 'left_hind_mid_leg',
    'left_hind_fetlock', 'left_hind_hoof', 'right_front_shoulder',
    'right_front_mid_leg', 'right_front_fetlock', 'right_front_hoof',
    'left_front_shoulder', 'left_front_mid_leg', 'left_front_fetlock',
    'left_front_hoof'
]


class TLEAPHandler:
    def __init__(self):
        self.s3 = boto3.client("s3")
        self.pose_model = None

        if YOLO_AVAILABLE:
            self._load_model()

    def _load_model(self):
        model_paths = [
            Path("/app/data/models/cow_pose_roboflow.pt"),
            Path("/app/shared/models/tleap/cow_pose.pt"),
        ]
        for p in model_paths:
            if p.exists():
                self.pose_model = YOLOPose(str(p))
                logger.info(f"Loaded pose model: {p}")
                return

        self.pose_model = YOLOPose("yolov8n-pose.pt")
        logger.info("Using generic YOLOv8 pose model")

    def handle(self, data: dict) -> dict:
        video_path = self._resolve_video(data)
        return self._extract_poses(video_path, data.get("video_id", "unknown"))

    def _resolve_video(self, data: dict) -> Path:
        source = data.get("s3_video_path") or data.get("video_path", "")
        if source.startswith("s3://"):
            parts = source.replace("s3://", "").split("/", 1)
            suffix = Path(parts[1]).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            self.s3.download_file(parts[0], parts[1], tmp.name)
            return Path(tmp.name)
        return Path(source)

    def _extract_poses(self, video_path: Path, video_id: str) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, fps // 2)
        pose_sequences = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0 and self.pose_model is not None:
                try:
                    results = self.pose_model(frame, verbose=False)
                    frame_poses = self._parse_pose_results(results, frame_count, fps)
                    if frame_poses:
                        pose_sequences.append(frame_poses)
                except Exception as e:
                    logger.warning(f"Pose estimation error at frame {frame_count}: {e}")

            frame_count += 1

        cap.release()

        locomotion = self._compute_locomotion_features(pose_sequences) if pose_sequences else {}

        return {
            "video_id": video_id,
            "pose_sequences": pose_sequences,
            "locomotion_features": locomotion,
            "total_frames": total_frames,
            "fps": fps,
            "frames_with_pose": len(pose_sequences)
        }

    def _parse_pose_results(self, results, frame_idx: int, fps: float) -> dict:
        for result in results:
            if not hasattr(result, 'keypoints') or result.keypoints is None:
                continue
            kps = result.keypoints
            if kps.xy is None or len(kps.xy) == 0:
                continue

            keypoints_xy = kps.xy[0].cpu().numpy()
            confs = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(len(keypoints_xy))

            kp_list = []
            for i, (xy, c) in enumerate(zip(keypoints_xy, confs)):
                name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"kp_{i}"
                kp_list.append({
                    "name": name, "x": float(xy[0]), "y": float(xy[1]),
                    "confidence": float(c)
                })

            return {
                "frame": frame_idx,
                "time": frame_idx / fps if fps > 0 else 0,
                "keypoints": kp_list,
                "num_keypoints_detected": sum(1 for kp in kp_list if kp["confidence"] > 0.3)
            }

        return {}

    def _compute_locomotion_features(self, pose_sequences: list) -> dict:
        if len(pose_sequences) < 2:
            return {"stride_regularity": 0, "symmetry_score": 0}

        hoof_positions = {"left_hind": [], "right_hind": [], "left_front": [], "right_front": []}

        for pose in pose_sequences:
            for kp in pose.get("keypoints", []):
                for limb in hoof_positions:
                    if f"{limb}_hoof" in kp["name"] and kp["confidence"] > 0.3:
                        hoof_positions[limb].append((kp["x"], kp["y"]))

        symmetry_scores = []
        for side in [("left_hind", "right_hind"), ("left_front", "right_front")]:
            left, right = hoof_positions[side[0]], hoof_positions[side[1]]
            if len(left) > 1 and len(right) > 1:
                l_range = max(p[1] for p in left) - min(p[1] for p in left)
                r_range = max(p[1] for p in right) - min(p[1] for p in right)
                if max(l_range, r_range) > 0:
                    symmetry_scores.append(1 - abs(l_range - r_range) / max(l_range, r_range))

        return {
            "stride_regularity": float(np.mean(symmetry_scores)) if symmetry_scores else 0,
            "symmetry_score": float(np.mean(symmetry_scores)) if symmetry_scores else 0,
            "num_frames_analyzed": len(pose_sequences)
        }
