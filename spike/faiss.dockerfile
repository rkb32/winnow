FROM --platform=linux/arm64 python:3.11-slim
RUN pip install --no-cache-dir faiss-cpu numpy
COPY faiss_check.py /faiss_check.py
RUN python /faiss_check.py
