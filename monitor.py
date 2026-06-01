"""
monitor.py — Logging and Monitoring utilities
Centralises log setup and model performance tracking across all pipeline steps.

AWS credentials for CloudWatch must be provided via environment variables:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=...
Never hardcode credentials here or in any config file.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """Configure root logger with a stdout stream handler and a persistent file handler."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def write_metrics(metrics: dict, metrics_file: str = "logs/metrics.json") -> None:
    """Append a timestamped metrics record to a JSON Lines file.

    Each line is a self-contained JSON object so the file can be tailed,
    grepped, and ingested by log aggregators without parsing the whole file.
    """
    parent = os.path.dirname(metrics_file) or "."
    Path(parent).mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logging.getLogger(__name__).info("Metrics appended → %s", metrics_file)


def publish_cloudwatch(metrics: dict, namespace: str, region: str) -> None:
    """Push numeric metrics to AWS CloudWatch as custom metrics.

    Credentials are read automatically from environment variables by boto3.
    Skips gracefully when boto3 is unavailable or credentials are not configured,
    so the pipeline can run locally without any AWS setup.
    """
    log = logging.getLogger(__name__)
    try:
        import boto3

        cw = boto3.client("cloudwatch", region_name=region)
        metric_data = [
            {
                "MetricName": k,
                "Value": float(v),
                "Unit": "None",
                "Timestamp": datetime.now(timezone.utc),
            }
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        ]
        if metric_data:
            cw.put_metric_data(Namespace=namespace, MetricData=metric_data)
            log.info(
                "Published %d metric(s) to CloudWatch namespace '%s'",
                len(metric_data),
                namespace,
            )
    except ImportError:
        log.warning("boto3 not installed — CloudWatch metrics skipped")
    except Exception as exc:
        log.warning("CloudWatch publish failed (check credentials/region): %s", exc)
