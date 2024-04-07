#include <string>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include "Eigen/Dense"
#include <tvm/runtime/registry.h>

using ConstEigenVectorArrayMap = Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;

extern "C"
TVM_DLL int CustomGroupnorm_arm(DLTensor* X, DLTensor* num_groups, 
    DLTensor* scale, DLTensor* bias, DLTensor* out, float eps) {
    // Write your code below
    assert(4 == X->ndim);
    assert(4 == out->ndim);

    const float *X_data = reinterpret_cast<const float*>(X->data);
    const float *num_groups_data = reinterpret_cast<const float*>(num_groups->data);
    const float *scale_data = reinterpret_cast<const float*>(scale->data);
    const float *bias_data = reinterpret_cast<const float*>(bias->data);
    int num_groups_i = int(num_groups_data[0]);
    float* out_data = reinterpret_cast<float*>(out->data);
    const int N = X->shape[0];
    const int64_t C = X->shape[1] / num_groups_i; // assume [N C*num_groups H W]  per the spec

    int64_t sample_size = 1;
    for (auto i = 2; i < X->ndim; ++i) {
      sample_size *= X->shape[i];
    }
    sample_size *= C;

    for (auto i = 0; i < N * num_groups_i; ++i) {
        ConstEigenVectorArrayMap Xi(X_data + sample_size * i, sample_size);
        const float Xi_mean = Xi.mean();
        const float squared_norm = (Xi - Xi_mean).matrix().squaredNorm();
        const float inv_stdev = 1.f / std::sqrt(squared_norm / sample_size + eps);
        EigenVectorArrayMap Yi(out + sample_size * i, sample_size);
        const float channel_scale = inv_stdev * scale_data[i % (C * num_groups_i)];
        const float channel_shift = bias_data[i % (C * num_groups_i)] - Xi_mean * channel_scale;
        Yi = Xi * channel_scale + channel_shift;
    }
    return 1;
}