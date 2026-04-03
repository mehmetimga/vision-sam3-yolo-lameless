"""
SageMaker Bridge Service
Runs on Fargate (CPU). Listens to NATS for video events and delegates
GPU inference to SageMaker async endpoints. Publishes results back to NATS.

Replaces the 8 GPU containers running on EC2 with a single lightweight
orchestrator + pay-per-use SageMaker inference.
"""
import asyncio
import os
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("sagemaker-bridge")

sys_path_added = False


def ensure_imports():
    global sys_path_added
    if not sys_path_added:
        import sys
        sys.path.insert(0, "/app")
        sys_path_added = True


class SageMakerBridge:
    """Bridges NATS events to SageMaker GPU inference."""

    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()

        ensure_imports()
        from shared.utils.nats_client import NATSClient
        from shared.utils.sagemaker_client import SageMakerInferenceClient

        self.nats_client = NATSClient(str(self.config_path))
        self.sagemaker_client = SageMakerInferenceClient(
            endpoint_name=os.getenv("SAGEMAKER_ENDPOINT_NAME"),
            io_bucket=os.getenv("SAGEMAKER_IO_BUCKET"),
            region=os.getenv("AWS_REGION", "us-west-2"),
        )

        self.results_base = Path("/app/data/results")
        for subdir in ["yolo", "sam3", "dinov3", "tleap"]:
            (self.results_base / subdir).mkdir(parents=True, exist_ok=True)

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {"nats": {"subjects": {}}}

    async def process_video(self, video_data: dict):
        """Handle a preprocessed video event by running all GPU pipelines via SageMaker."""
        video_id = video_data["video_id"]
        processed_path = Path(video_data["processed_path"])

        logger.info(f"Processing video {video_id} via SageMaker")

        if not processed_path.exists():
            logger.error(f"Video not found: {processed_path}")
            return

        try:
            s3_video_path = self.sagemaker_client.upload_video_to_s3(
                str(processed_path), video_id
            )

            logger.info(f"Invoking SageMaker for all pipelines on {video_id}")
            results = self.sagemaker_client.invoke_all_pipelines(
                video_id=video_id,
                s3_video_path=s3_video_path,
                poll_interval=15.0,
                timeout=900.0,
            )

            await self._process_results(video_id, results, video_data)
            logger.info(f"All pipelines completed for {video_id}")

        except Exception as e:
            logger.error(f"SageMaker inference failed for {video_id}: {e}")
            import traceback
            traceback.print_exc()

    async def _process_results(self, video_id: str, results: dict, original_data: dict):
        """Save results to EFS and publish to NATS."""
        subjects = self.config.get("nats", {}).get("subjects", {})

        pipeline_configs = [
            ("yolo", "pipeline_yolo", ["features", "num_detections", "total_frames"]),
            ("sam3", "pipeline_sam3", ["aggregated_features", "num_segmentations"]),
            ("dinov3", "pipeline_dinov3", ["neighbor_evidence", "similar_cases", "embedding_dim"]),
            ("tleap", "pipeline_tleap", ["locomotion_features", "frames_with_pose"]),
        ]

        for pipeline_name, nats_subject_key, extra_fields in pipeline_configs:
            pipeline_results = results.get(pipeline_name)
            if not pipeline_results:
                logger.warning(f"No results for {pipeline_name}")
                continue

            results_file = self.results_base / pipeline_name / f"{video_id}_{pipeline_name}.json"
            with open(results_file, "w") as f:
                json.dump(pipeline_results, f, indent=2)

            nats_msg = {
                "video_id": video_id,
                "pipeline": pipeline_name,
                "results_path": str(results_file),
            }
            for field in extra_fields:
                if field in pipeline_results:
                    nats_msg[field] = pipeline_results[field]

            subject = subjects.get(nats_subject_key, f"pipeline.{pipeline_name}")
            await self.nats_client.publish(subject, nats_msg)
            logger.info(f"Published {pipeline_name} results to {subject}")

    async def start(self):
        """Start the bridge service."""
        await self.nats_client.connect()

        subject = self.config["nats"]["subjects"]["video_preprocessed"]
        logger.info(f"SageMaker bridge subscribed to {subject}")
        await self.nats_client.subscribe(subject, self.process_video)

        logger.info("SageMaker bridge service started. Waiting for videos...")
        await asyncio.Event().wait()


async def main():
    bridge = SageMakerBridge()
    await bridge.start()


if __name__ == "__main__":
    asyncio.run(main())
