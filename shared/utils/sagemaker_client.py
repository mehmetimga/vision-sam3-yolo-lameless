"""
SageMaker async inference client.
Handles submitting requests and polling for results from SageMaker endpoints.
"""
import os
import json
import time
import uuid
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class SageMakerInferenceClient:
    """Client for invoking SageMaker async inference endpoints."""

    def __init__(
        self,
        endpoint_name: Optional[str] = None,
        io_bucket: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.region = region or os.getenv("AWS_REGION", "us-west-2")
        self.endpoint_name = endpoint_name or os.getenv("SAGEMAKER_ENDPOINT_NAME")
        self.io_bucket = io_bucket or os.getenv("SAGEMAKER_IO_BUCKET")

        self.sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=self.region)
        self.s3 = boto3.client("s3", region_name=self.region)

    def invoke(self, request_data: dict, poll_interval: float = 10.0, timeout: float = 900.0) -> dict:
        """
        Submit an async inference request and wait for the result.

        Args:
            request_data: JSON-serializable request (must include 'pipeline' and 'video_id')
            poll_interval: Seconds between polling attempts
            timeout: Maximum seconds to wait for result

        Returns:
            Inference result as a dict
        """
        request_id = str(uuid.uuid4())
        input_key = f"input/{request_id}.json"

        self.s3.put_object(
            Bucket=self.io_bucket,
            Key=input_key,
            Body=json.dumps(request_data),
            ContentType="application/json"
        )

        input_location = f"s3://{self.io_bucket}/{input_key}"
        logger.info(f"Invoking SageMaker endpoint={self.endpoint_name}, request_id={request_id}")

        response = self.sagemaker_runtime.invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            InputLocation=input_location,
            ContentType="application/json",
        )

        output_location = response.get("OutputLocation", "")
        logger.info(f"Waiting for result at {output_location}")

        return self._poll_for_result(output_location, poll_interval, timeout)

    def invoke_all_pipelines(
        self, video_id: str, s3_video_path: str,
        poll_interval: float = 10.0, timeout: float = 900.0
    ) -> dict:
        """Run all GPU pipelines (YOLO, SAM3, DINOv3, T-LEAP) in a single request."""
        return self.invoke(
            request_data={
                "pipeline": "all",
                "video_id": video_id,
                "s3_video_path": s3_video_path,
            },
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def invoke_pipeline(
        self, pipeline: str, video_id: str, s3_video_path: str,
        extra_data: Optional[dict] = None,
        poll_interval: float = 10.0, timeout: float = 900.0
    ) -> dict:
        """Run a single pipeline."""
        data = {
            "pipeline": pipeline,
            "video_id": video_id,
            "s3_video_path": s3_video_path,
        }
        if extra_data:
            data.update(extra_data)
        return self.invoke(data, poll_interval, timeout)

    def _poll_for_result(self, output_location: str, poll_interval: float, timeout: float) -> dict:
        """Poll S3 for the inference result."""
        if not output_location.startswith("s3://"):
            raise ValueError(f"Invalid output location: {output_location}")

        parts = output_location.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.s3.get_object(Bucket=bucket, Key=key)
                body = response["Body"].read().decode("utf-8")
                logger.info(f"Result received after {time.time() - start_time:.1f}s")
                return json.loads(body)
            except ClientError as e:
                if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                    elapsed = time.time() - start_time
                    if elapsed > 300:
                        logger.info(f"Still waiting... ({elapsed:.0f}s elapsed, cold start may be in progress)")
                    time.sleep(poll_interval)
                else:
                    raise

        raise TimeoutError(
            f"SageMaker inference timed out after {timeout}s. "
            "The endpoint may still be scaling up from zero."
        )

    def upload_video_to_s3(self, local_path: str, video_id: str) -> str:
        """Upload a video file from local/EFS to S3 for SageMaker processing."""
        s3_key = f"videos/{video_id}/{os.path.basename(local_path)}"
        self.s3.upload_file(local_path, self.io_bucket, s3_key)
        s3_uri = f"s3://{self.io_bucket}/{s3_key}"
        logger.info(f"Uploaded video to {s3_uri}")
        return s3_uri
