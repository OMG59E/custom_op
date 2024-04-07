import torch
import onnxruntime as ort


session_options = ort.SessionOptions()
session_options.register_custom_ops_library("build/libcustom_group_norm.so")

m = ort.InferenceSession("../pytorch_custom_op/model.onnx", session_options)

X = torch.randn(3, 2, 1, 2, dtype=torch.float32).numpy()
num_groups = torch.tensor([2.], dtype=torch.float32).numpy()
scale = torch.tensor([1., 1.], dtype=torch.float32).numpy()
bias = torch.tensor([0., 0.], dtype=torch.float32).numpy()

datas = {
    "X": X,
    "num_groups": num_groups,
    "scale": scale,
    "bias": bias
    }

res = m.run(None, datas)
print(res[0].flatten())