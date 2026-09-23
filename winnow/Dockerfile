# The embedder file is CPU-independent, so build it natively instead of under ARM emulation.
FROM --platform=$BUILDPLATFORM python:3.11-slim AS model
RUN pip install --no-cache-dir onnx==1.23.0
ADD --checksum=sha256:c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5 \
    https://huggingface.co/opencv/image_classification_mobilenet/resolve/main/image_classification_mobilenetv2_2022apr.onnx \
    /models/mobilenetv2.onnx
COPY models/make_embedder.py /models/
RUN python /models/make_embedder.py /models/mobilenetv2.onnx /models/embedder.onnx

FROM python:3.11-slim
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
COPY --from=model /models/embedder.onnx /app/models/embedder.onnx
COPY duplicates.py blur.py exif_conflict.py semantic.py scan.py s3_scan.py agent.py /app/
