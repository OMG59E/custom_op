#include <string>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <tvm/runtime/registry.h>
#include "torch/script.h"

torch::Tensor DLTensorToTorchTensor(DLTensor* x) {
    torch::TensorOptions option(torch::kFloat32);
    return torch::from_blob(x->data, {x->shape[0], x->shape[1], x->shape[2], x->shape[3]}, option);
}

extern "C"
TVM_DLL int EdgexResizeNearest_arm(DLTensor* x, DLTensor* y, float scale_factor) {
    torch::Tensor _x = DLTensorToTorchTensor(x);
    torch::Tensor _y = DLTensorToTorchTensor(y);
    auto tmp = torch::upsample_nearest2d(_x, {(int)(_x.size(2) * scale_factor), (int)(_x.size(3) * scale_factor)});
    _y.copy_(tmp);
    return 1;
}