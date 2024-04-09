#include <string>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <tvm/runtime/registry.h>
#include "ATen/ops/dropout.h"

extern "C"
TVM_DLL int EdgexDropout_arm(DLTensor* input, DLTensor* output, DLTensor* mask, float p) {

    at::Tensor _input;
    at::Tensor dropout(_input, p, true);
    return 1;
}