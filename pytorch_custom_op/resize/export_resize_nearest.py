import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.onnx import OperatorExportTypes, TrainingMode


class CustomResizeNearestOp(Function):
    @staticmethod
    def forward(ctx, x, scale_factor):
        y = F.upsample(x, scale_factor=scale_factor, mode="nearest")
        return y
    
    @staticmethod
    def symbolic(g, x,scale_factor):
        return g.op("EdgexResizeNearest", x, scale_factor_f=scale_factor)
    
    
class EdgexResizeNearest(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = CustomResizeNearestOp.apply
        
    def forward(self, x):
        y = self.op(x, 0.5)
        return y
    

m = EdgexResizeNearest()
m.eval()

x = torch.randn(1, 3, 4, 4, dtype=torch.float32)
torch.onnx.export(m, x, "resize.onnx", 
                  opset_version=15,
                  verbose=True, 
                  input_names=["x"], 
                  output_names=["y"],
                  operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK
                  )
