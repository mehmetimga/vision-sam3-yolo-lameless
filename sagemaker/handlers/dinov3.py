"""DINOv3 embedding handler for SageMaker endpoint."""
import tempfile
import logging
from pathlib import Path

import boto3
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

logger = logging.getLogger("sagemaker-serve.dinov3")


class DINOv3Handler:
    def __init__(self):
        self.s3 = boto3.client("s3")
        model_name = "facebook/dinov2-base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading DINOv3 model {model_name} on {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        logger.info("DINOv3 model loaded")

    def handle(self, data: dict) -> dict:
        video_path = self._resolve_video(data)
        return self._extract_embeddings(video_path, data.get("video_id", "unknown"))

    def _resolve_video(self, data: dict) -> Path:
        source = data.get("s3_video_path") or data.get("video_path", "")
        if source.startswith("s3://"):
            parts = source.replace("s3://", "").split("/", 1)
            suffix = Path(parts[1]).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            self.s3.download_file(parts[0], parts[1], tmp.name)
            return Path(tmp.name)
        return Path(source)

    def _embed_frame(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        pil_image = Image.fromarray(frame_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embedding

    def _extract_embeddings(self, video_path: Path, video_id: str) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, fps)
        embeddings = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                emb = self._embed_frame(frame)
                embeddings.append({
                    "frame": frame_count,
                    "time": frame_count / fps if fps > 0 else 0,
                    "embedding": emb.tolist()
                })

            frame_count += 1

        cap.release()

        avg_embedding = []
        if embeddings:
            avg_embedding = np.mean(
                [np.array(e["embedding"]) for e in embeddings], axis=0
            ).tolist()

        canonical = []
        if embeddings:
            canonical = [embeddings[0], embeddings[len(embeddings) // 2], embeddings[-1]]

        return {
            "video_id": video_id,
            "embedding_dim": len(avg_embedding),
            "num_embeddings": len(embeddings),
            "avg_embedding": avg_embedding,
            "canonical_frames": canonical,
            "total_frames": total_frames,
            "fps": fps
        }
