# store-sales-mlops
End-to-end LightGBM sales forecasting pipeline deployed on AWS — ECS training, S3 storage, Lambda inference, and CloudWatch monitoring.

## Architecture

- **S3** — raw data, processed features, trained model artifacts, and predictions
- **ECS (Fargate)** — runs the training pipeline (preprocess → features → train)
- **ECR** — stores Docker images for both training and inference
- **Lambda (`store-sales-trigger`)** — event-driven trigger: fires on S3 upload, starts ECS retraining
- **Lambda (`store-sales-inference`)** — on-demand inference endpoint, loads model from S3 *(optional inference path, not in final presentation)*
- **API Gateway** — HTTP API exposing `POST /predict` *(optional inference path, not in final presentation)*
- **CloudWatch** — logs and model performance metrics

### Deployment patterns
| Pattern | Trigger | What happens |
|---|---|---|
| Event-driven retraining | Upload CSV to `s3://.../uploads/` | S3 → trigger Lambda → ECS reruns full pipeline |
| On-demand inference *(optional inference path, not in final presentation)* | `POST /predict` | API Gateway → inference Lambda → predictions saved to S3 |

## AWS Deployment

### Prerequisites
- AWS CLI configured for your account
- Docker Desktop running

### 1. Upload raw data to S3
The pipeline expects these raw CSV files in the S3 `data_dir` bucket (configured in `config.yaml`):
- `train.csv`
- `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `transactions.csv`

Upload them with:
```bash
aws s3 cp train.csv s3://mlds423-finalproject-s3-061/
aws s3 cp test.csv s3://mlds423-finalproject-s3-061/
aws s3 cp stores.csv s3://mlds423-finalproject-s3-061/
aws s3 cp oil.csv s3://mlds423-finalproject-s3-061/
aws s3 cp holidays_events.csv s3://mlds423-finalproject-s3-061/
aws s3 cp transactions.csv s3://mlds423-finalproject-s3-061/
```

### 2. Build and push the training image
> All services deploy to **us-east-2**.
```bash
aws ecr create-repository --repository-name store-sales-pipeline --region us-east-2

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com

docker build --platform linux/amd64 -t store-sales-pipeline .
docker tag store-sales-pipeline:latest <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-pipeline:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-pipeline:latest
```

### 3. Build and push the inference image
```bash
aws ecr create-repository --repository-name store-sales-inference --region us-east-2

docker buildx build --provenance=false --platform linux/amd64 -f Dockerfile.lambda -t store-sales-inference .
docker tag store-sales-inference:latest <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-inference:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-inference:latest
```

### 4. IAM roles
Create two IAM roles in the AWS console (IAM → Roles → Create role → Elastic Container Service Task):

**ecsTaskExecutionRole** — attach policy:
- `AmazonECSTaskExecutionRolePolicy`

**ecsTaskRole** — attach policies:
- `AmazonS3FullAccess`
- `CloudWatchFullAccess`

Update the trust policy of `ecsTaskRole` to also allow Lambda:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": ["ecs-tasks.amazonaws.com", "lambda.amazonaws.com"]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### 5. ECS training pipeline
1. Create cluster: ECS → Clusters → Create → Fargate → name `store-sales-cluster`
2. Create task definition: ECS → Task Definitions → Create
   - Launch type: Fargate
   - Task execution role: `ecsTaskExecutionRole`
   - Task role: `ecsTaskRole`
   - CPU: 2 vCPU, Memory: 8 GB
   - Container image: `<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-pipeline:latest`
3. Run task: Clusters → store-sales-cluster → Tasks → Run new task → Fargate

### 6. Lambda inference endpoint ⚠️ Optional Inference Path — not included in final presentation
> This section sets up an on-demand inference endpoint via API Gateway. It is fully functional but was not presented as part of the final project submission. Included here for reference only.

1. Lambda → Create function → Container image
   - Name: `store-sales-inference`
   - Image: `<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-inference:latest`
   - Execution role: `ecsTaskRole`
2. Set timeout to 5 min and memory to 2048 MB

### 7. API Gateway ⚠️ Optional Inference Path — not included in final presentation
> Exposes the inference Lambda as a public HTTP endpoint. Functional but not part of the final project submission.

1. API Gateway → Create API → HTTP API
2. Integration: Lambda → `store-sales-inference`
3. Route: `POST /predict`
4. Deploy

### 8. Event-driven retraining trigger
This Lambda fires whenever a CSV is uploaded to S3, automatically triggering the appropriate ECS pipeline.

**Create the trigger Lambda:**
1. Lambda → Create function → Author from scratch
   - Name: `store-sales-trigger`
   - Runtime: Python 3.11
3. Paste the code from `lambda_trigger.py`
4. Go to **Code → Runtime settings → Edit** → set Handler to `lambda_function.handler`
5. Click **Deploy**

**Grant ECS permissions to the trigger Lambda:**
1. Lambda → Configuration → Permissions → click the execution role
2. IAM → Add permissions → Create inline policy → JSON:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["ecs:RunTask", "iam:PassRole"],
      "Resource": "*"
    }
  ]
}
```
3. Name it `ecs-runtask` → Create policy

**Configure S3 event notification:**
1. S3 → mlds423-finalproject-s3-061 → Properties → Event notifications → Create
   - Name: `trigger-pipeline`
   - Prefix: (empty)
   - Suffix: (empty)
   - Event type: **All object create events** (`s3:ObjectCreated:*`)
   - Destination: Lambda → `store-sales-trigger`
2. Save

> **Event routing logic (`lambda_trigger.py`):** The trigger Lambda inspects the filename of the uploaded object and routes to the appropriate ECS pipeline:
> - Filename ends with `train.csv` → ECS runs `preprocess.py → features.py → train.py` (full retraining)
> - Filename ends with `test.csv` → ECS runs `preprocess.py → features.py → predict.py` (inference, predictions saved to S3)
> - Any other filename → ignored, no ECS task started

> Use `s3:ObjectCreated:*` (not just PUT) to ensure multipart uploads (used by `aws s3 cp` for large files) also trigger the notification.

**Trigger a retrain:**
```bash
aws s3api put-object --bucket mlds423-finalproject-s3-061 --key train.csv --body train.csv
```

**Trigger inference on new test data:**
```bash
aws s3api put-object --bucket mlds423-finalproject-s3-061 --key test.csv --body test.csv
```

Check ECS → Clusters → store-sales-cluster → Tasks to see the task start automatically.

### Troubleshooting: Lambda 403 on S3
If the Lambda function returns `403 Forbidden` when accessing S3 despite having `AmazonS3FullAccess`, add an explicit inline policy directly to `ecsTaskRole`:

1. IAM → Roles → ecsTaskRole → Add permissions → Create inline policy → JSON:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::mlds423-finalproject-s3-061",
        "arn:aws:s3:::mlds423-finalproject-s3-061/*"
      ]
    }
  ]
}
```
2. Name it `s3-bucket-access` → Create policy

### Running inference
Use default paths from `config.yaml` (runs on `features_test.parquet` already in S3):
```bash
curl -X POST https://<API_ID>.execute-api.us-east-2.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

Or pass a custom pre-engineered features parquet:
```bash
curl -X POST https://<API_ID>.execute-api.us-east-2.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"input_path": "s3://mlds423-finalproject-s3-061/processed/my-features.parquet"}'
```

> **Note:** `input_path` must be a pre-engineered parquet file (output of `features.py`) with lag features and rolling stats already computed — not a raw CSV.

Returns:
```json
{
  "message": "Inference complete",
  "n_predictions": 28512,
  "output_path": "s3://mlds423-finalproject-s3-061/processed/predictions.csv"
}
```

## Project structure
```
.
├── Dockerfile              # Training pipeline image
├── Dockerfile.lambda       # Inference Lambda image
├── config.yaml             # Pipeline configuration (paths, hyperparameters)
├── preprocess.py           # Step 1: data cleaning
├── features.py             # Step 2: feature engineering
├── train.py                # Step 3: model training, saves artifact to S3
├── predict.py              # Step 4: inference, loads model from S3
├── monitor.py              # CloudWatch metrics + logging utilities
├── lambda_handler.py       # Inference Lambda entry point wrapping predict.py
├── lambda_trigger.py       # Trigger Lambda: responds to S3 uploads, starts ECS task
└── requirements.txt
```

## Configuration
All pipeline parameters are in `config.yaml`. AWS credentials must be set via environment variables — never hardcoded:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```
