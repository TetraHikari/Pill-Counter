# Model server (src/api.py). The models and thresholds are not baked in: docker-compose.yml mounts results/.
FROM python:3.13-slim

# OpenCV (pulled in by ultralytics) needs these system libraries.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch first (Docker on a Mac can't use the Apple GPU; the default x86 wheel bundles CUDA, ~3 GB more).
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
ENV YOLO_CONFIG_DIR=/tmp/ultralytics PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
