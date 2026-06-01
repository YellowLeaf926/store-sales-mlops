FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY monitor.py .
COPY preprocess.py .
COPY features.py .
COPY train.py .
COPY predict.py .

CMD ["python3", "preprocess.py"]