# store-sales-mlops
End-to-end LightGBM sales forecasting pipeline deployed on AWS — ECS training, S3 storage, Lambda inference, and CloudWatch monitoring.

## Architecture

- **S3** — raw data, processed features, trained model artifacts, and predictions
- **ECS (Fargate)** — runs the training pipeline (preprocess → features → train)
- **ECR** — stores Docker images for both training and inference
- **Lambda** — on-demand inference endpoint, loads model from S3
- **API Gateway** — HTTP API exposing `POST /predict`
- **CloudWatch** — logs and model performance metrics

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
```bash
aws ecr create-repository --repository-name store-sales-pipeline --region us-east-2

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com

docker build --platform linux/amd64 -t store-sales-pipeline .
docker tag store-sales-pipeline:latest <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-pipeline:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-pipeline:latest
```

### 2. Build and push the inference image
```bash
aws ecr create-repository --repository-name store-sales-inference --region us-east-2

docker buildx build --provenance=false --platform linux/amd64 -f Dockerfile.lambda -t store-sales-inference .
docker tag store-sales-inference:latest <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-inference:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-inference:latest
```

### 3. IAM roles
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

### 4. ECS training pipeline
1. Create cluster: ECS → Clusters → Create → Fargate → name `store-sales-cluster`
2. Create task definition: ECS → Task Definitions → Create
   - Launch type: Fargate
   - Task execution role: `ecsTaskExecutionRole`
   - Task role: `ecsTaskRole`
   - CPU: 2 vCPU, Memory: 8 GB
   - Container image: `<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-pipeline:latest`
3. Run task: Clusters → store-sales-cluster → Tasks → Run new task → Fargate

### 5. Lambda inference endpoint
1. Lambda → Create function → Container image
   - Name: `store-sales-inference`
   - Image: `<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/store-sales-inference:latest`
   - Execution role: `ecsTaskRole`
2. Set timeout to 5 min and memory to 2048 MB

### 6. API Gateway
1. API Gateway → Create API → HTTP API
2. Integration: Lambda → `store-sales-inference`
3. Route: `POST /predict`
4. Deploy

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
```bash
curl -X POST https://<API_ID>.execute-api.us-east-2.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

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
├── lambda_handler.py       # Lambda entry point wrapping predict.py
└── requirements.txt
```

## Configuration
All pipeline parameters are in `config.yaml`. AWS credentials must be set via environment variables — never hardcoded:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```
