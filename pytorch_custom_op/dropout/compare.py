import torch
import onnxruntime as ort
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        y = self.dropout(x)
        return y
    
mt = Dropout()
x = torch.randn(1, 3, 3, 3, dtype=torch.float32).cpu()
y = mt(x)
print(y.numpy().flatten())

sess_options = ort.SessionOptions()
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
mo = ort.InferenceSession("dropout.onnx", disabled_optimizers=["EliminateDropout"])
print(x.numpy().flatten())
y = mo.run(None, {"x": x.numpy()})[0]
print(y.flatten())