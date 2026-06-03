import json
import logging

import yaml

from predict import run

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def handler(event, context):
    cfg = yaml.safe_load(open("config.yaml"))

    body = {}
    if event.get("body"):
        try:
            body = json.loads(event["body"])
        except Exception:
            pass

    model_path = body.get("model_path", cfg["paths"]["model_dir"] + "lgb_model.pkl")
    input_path = body.get("input_path", cfg["paths"]["output_dir"] + "features_test.parquet")
    output_path = body.get("output_path", cfg["paths"]["output_dir"] + "predictions.csv")

    monitoring_cfg = {
        "metrics_file": "/tmp/metrics.json",
        "cloudwatch": cfg["monitoring"]["cloudwatch"],
    }

    try:
        result = run(model_path, input_path, output_path, monitoring_cfg)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "message": "Inference complete",
                "n_predictions": len(result),
                "output_path": output_path,
            }),
        }
    except Exception as e:
        log.error("Inference failed: %s", str(e))
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
