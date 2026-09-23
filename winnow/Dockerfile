FROM python:3.11-slim

RUN pip install --no-cache-dir "opencv-contrib-python-headless>=5,<6" numpy Pillow boto3

WORKDIR /app
COPY duplicates.py blur.py exif_conflict.py scan.py s3_scan.py agent.py /app/
