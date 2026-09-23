"""Turn the OpenCV Zoo MobileNetV2 classifier into an image embedder.

OpenCV 5's DNN engine can only return a model's declared outputs, so instead of
asking for an intermediate layer at runtime, we drop the final classifier and make
its 1280-d input (the pooled image features) the model's output.
"""
import sys

import onnx
from onnx import TensorProto, helper

src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)
graph = model.graph

classifier = graph.node[-1]
assert classifier.op_type == "Gemm", classifier.op_type
features, weight, bias = classifier.input

graph.node.remove(classifier)
for init in [i for i in graph.initializer if i.name in (weight, bias)]:
    graph.initializer.remove(init)
del graph.output[:]
graph.output.append(helper.make_tensor_value_info(features, TensorProto.FLOAT, [1, 1280]))

onnx.checker.check_model(model)
onnx.save(model, dst)
