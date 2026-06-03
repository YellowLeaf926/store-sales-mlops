FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY monitor.py .
COPY preprocess.py .
COPY features.py .
COPY train.py .
COPY predict.py .

CMD ["sh", "-c", "python3 preprocess.py && python3 features.py && python3 train.py"]
