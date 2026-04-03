"""
SageMaker inference server for GPU vision pipelines.
Serves YOLO, SAM3, DINOv3, and T-LEAP models behind a single endpoint.
Routes to the appropriate handler based on the 'pipeline' field in the request.
"""
import os
import sys
import json
import logging
import traceback
from pathlib import Path

from flask import Flask, request, jsonify

sys.path.insert(0, "/app")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("sagemaker-serve")

app = Flask(__name__)

handlers = {}


def load_handlers():
    """Lazily load model handlers on first request to each pipeline."""
    pass


def get_handler(pipeline_name: str):
    """Get or initialize a handler for the given pipeline."""
    if pipeline_name in handlers:
        return handlers[pipeline_name]

    logger.info(f"Loading handler for pipeline: {pipeline_name}")

    if pipeline_name == "yolo":
        from sagemaker.handlers.yolo import YOLOHandler
        handlers[pipeline_name] = YOLOHandler()
    elif pipeline_name == "sam3":
        from sagemaker.handlers.sam3 import SAM3Handler
        handlers[pipeline_name] = SAM3Handler()
    elif pipeline_name == "dinov3":
        from sagemaker.handlers.dinov3 import DINOv3Handler
        handlers[pipeline_name] = DINOv3Handler()
    elif pipeline_name == "tleap":
        from sagemaker.handlers.tleap import TLEAPHandler
        handlers[pipeline_name] = TLEAPHandler()
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    logger.info(f"Handler loaded for {pipeline_name}")
    return handlers[pipeline_name]


@app.route("/ping", methods=["GET"])
def ping():
    return "", 200


@app.route("/invocations", methods=["POST"])
def invoke():
    try:
        if request.content_type != "application/json":
            return jsonify({"error": f"Unsupported content type: {request.content_type}"}), 415

        data = request.get_json()
        pipeline = data.get("pipeline", "yolo")
        video_id = data.get("video_id", "unknown")

        logger.info(f"Inference request: pipeline={pipeline}, video_id={video_id}")

        if pipeline == "all":
            results = _run_all_pipelines(data)
        else:
            handler = get_handler(pipeline)
            results = handler.handle(data)

        logger.info(f"Inference complete: pipeline={pipeline}, video_id={video_id}")
        return jsonify(results)

    except Exception as e:
        logger.error(f"Inference error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


def _run_all_pipelines(data: dict) -> dict:
    """Run all GPU pipelines sequentially on the same video."""
    video_id = data.get("video_id", "unknown")
    results = {"video_id": video_id}

    yolo_handler = get_handler("yolo")
    yolo_results = yolo_handler.handle(data)
    results["yolo"] = yolo_results

    sam3_data = {**data, "yolo_results": yolo_results}
    sam3_handler = get_handler("sam3")
    results["sam3"] = sam3_handler.handle(sam3_data)

    dinov3_handler = get_handler("dinov3")
    results["dinov3"] = dinov3_handler.handle(data)

    tleap_handler = get_handler("tleap")
    results["tleap"] = tleap_handler.handle(data)

    return results


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
