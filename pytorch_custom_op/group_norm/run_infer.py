import torch

torch.ops.load_library("build/lib.linux-x86_64-cpython-38/custom_group_norm.cpython-38-x86_64-linux-gnu.so")


class CustomModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.custom_group_norm = torch.ops.mynamespace.custom_group_norm
    def forward(self, x, num_groups, scale, bias):
        return self.custom_group_norm(x, num_groups, scale, bias, 0.)
    
m = CustomModel()
m.eval()

X = torch.randn(3, 2, 1, 2, dtype=torch.float32)
num_groups = torch.tensor([2.], dtype=torch.float32)
scale = torch.tensor([1., 1.], dtype=torch.float32)
bias = torch.tensor([0., 0.], dtype=torch.float32)
inputs = (X, num_groups, scale, bias)

y = m(*inputs)
print(y.numpy().flatten())
