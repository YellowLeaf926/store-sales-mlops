import boto3

ECS_CLUSTER = "store-sales-cluster"
ECS_TASK_DEF = "store-sales-task"
SUBNET_ID = "subnet-834eb1fe"
REGION = "us-east-2"

TRAINING_CMD = "python3 preprocess.py && python3 features.py && python3 train.py"
INFERENCE_CMD = "python3 preprocess.py && python3 features.py && python3 predict.py"


def handler(event, context):
    record = event["Records"][0]["s3"]
    bucket = record["bucket"]["name"]
    key = record["object"]["key"]

    ecs = boto3.client("ecs", region_name=REGION)

    if key.endswith("train.csv"):
        cmd = TRAINING_CMD
        pipeline = "training"
    elif key.endswith("test.csv"):
        cmd = INFERENCE_CMD
        pipeline = "inference"
    else:
        print(f"Ignoring upload: s3://{bucket}/{key}")
        return {"statusCode": 200, "body": "No action taken"}

    response = ecs.run_task(
        cluster=ECS_CLUSTER,
        taskDefinition=ECS_TASK_DEF,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": [SUBNET_ID],
                "assignPublicIp": "ENABLED"
            }
        },
        overrides={
            "containerOverrides": [{
                "name": "store-sales-container",
                "command": ["sh", "-c", cmd]
            }]
        }
    )

    task_arn = response["tasks"][0]["taskArn"]
    print(f"Started {pipeline} ECS task {task_arn} for s3://{bucket}/{key}")
    return {"statusCode": 200, "body": f"{pipeline} task started: {task_arn}"}
