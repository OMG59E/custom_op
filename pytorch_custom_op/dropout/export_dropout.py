import torch
import torch.nn as nn
from torch.autograd import Function
from torch.onnx import OperatorExportTypes, TrainingMode



def dropout_forward(input, p=0.5, training=True):
    if not training or p == 0:
        # 训练模式未启用或p为0时，直接返回输入
        return input, None
    # 计算保留概率（1-p）和相应的缩放因子
    p1m = 1.0 - p
    scale = 1.0 / p1m

    # 生成和输入相同形状的mask，数据类型为bool
    mask = torch.empty_like(input, dtype=torch.bool)

    # 对mask应用伯努利分布采样，采样概率为保留概率（1-p）
    mask.bernoulli_(p1m)

    # 应用dropout
    output = input * mask * scale
    return output, mask


class CustomDropoutOp(Function):
    @staticmethod
    def forward(ctx, x, p):
        y, mask = dropout_forward(x, p=p, training=True)
        return y, mask
    
    @staticmethod
    def symbolic(g, x, p):
        return g.op("EdgexDropout", x, p_f=p, outputs=2)
    
    
class EdgexDropout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = CustomDropoutOp.apply
        
    def forward(self, x):
        y, mask = self.dropout(x, 0.5)
        return y, mask
    

m = EdgexDropout()
m.eval()

x = torch.randn(1, 3, 3, 3, dtype=torch.float32)
torch.onnx.export(m, x, "dropout_forward.onnx", 
                  opset_version=15,
                  verbose=True, 
                  input_names=["input"], 
                  output_names=["output", "mask"],
                  operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK
                  )
