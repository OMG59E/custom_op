# SPDX-License-Identifier: Apache-2.0

import torch


def register_custom_op():
    def my_group_norm(g, input, num_groups, scale, bias, eps):
        return g.op("mydomain::CustomGroupnorm", input, num_groups, scale, bias, epsilon_f=0.)

    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic("mynamespace::custom_group_norm", my_group_norm, opset_version=13)


def export_custom_op():
    class CustomModel(torch.nn.Module):
        def forward(self, x, num_groups, scale, bias):
            return torch.ops.mynamespace.custom_group_norm(x, num_groups, scale, bias, 0.)
    m = CustomModel()
    m.eval()
    
    X = torch.randn(3, 2, 1, 2)
    num_groups = torch.tensor([2.])
    scale = torch.tensor([1., 1.])
    bias = torch.tensor([0., 0.])
    inputs = (X, num_groups, scale, bias)

    f = './model.onnx'
    torch.onnx.export(m, inputs, f,
                      input_names=["X", "num_groups", "scale", "bias"], output_names=["Y"],
                      custom_opsets={"mydomain": 1},
                      opset_version=13
                      )


torch.ops.load_library("build/lib.linux-x86_64-cpython-38/custom_group_norm.cpython-38-x86_64-linux-gnu.so")
register_custom_op()
export_custom_op()
