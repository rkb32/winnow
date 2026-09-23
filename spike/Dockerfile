FROM python:3.11-slim

RUN pip install --no-cache-dir "opencv-contrib-python-headless>=5,<6" numpy

COPY test_imghash.py /test_imghash.py

CMD ["python", "/test_imghash.py"]
