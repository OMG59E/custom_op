/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <mutex>
#include "custom_op.h"
#include "Eigen/Dense"
#include "onnxruntime_cxx_api.h"

template <typename T>
using ConstEigenVectorArrayMap = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
void GroupNormKernel<T>::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const T* X_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_X));
  const OrtValue* input_num_groups = ort_.KernelContext_GetInput(context, 1);
  const T* num_groups = reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_num_groups));
  const OrtValue* input_scale = ort_.KernelContext_GetInput(context, 2);
  const T* scale_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_scale));
  const OrtValue* input_B = ort_.KernelContext_GetInput(context, 3);
  const T* B_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_B));

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);
  const int64_t N = dimensions[0];
  const int64_t C = dimensions[1] / num_groups[0];  // assume [N C*num_groups H W]  per the spec

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
	int64_t sample_size = 1;
  for (size_t i = 2; i < dimensions.size(); ++i) {
    sample_size *= dimensions[i];
  }
  sample_size *= C;

  for (auto i = 0; i < N * num_groups[0]; ++i) {
    ConstEigenVectorArrayMap<float> Xi(X_data + sample_size * i, sample_size);
    const float Xi_mean = Xi.mean();
    const float squared_norm = (Xi - Xi_mean).matrix().squaredNorm();
    const float inv_stdev = 1.0f / std::sqrt(squared_norm / sample_size + epsilon_);
    EigenVectorArrayMap<float> Yi(out + sample_size * i, sample_size);
    const float channel_scale = inv_stdev * scale_data[i % (C * int(num_groups[0]))];
    const float channel_shift = B_data[i % (C * int(num_groups[0]))] - Xi_mean * channel_scale;
    Yi = Xi * channel_scale + channel_shift;
  }
}



struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi* ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain* domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi* ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain, const OrtApi* ort_api) {
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  // GroupNormCustomOp custom_op;
  // Ort::CustomOpDomain custom_op_domain("mydomain");
  // custom_op_domain.Add(&custom_op);

  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain("mydomain", &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);
  printf("1111111111111\n");
  GroupNormCustomOp custom_op;
  if (auto status = ortApi->CustomOpDomain_Add(domain, &custom_op)) {
    return status;
  }
  printf("222222222222222\n");
  return ortApi->AddCustomOpDomain(options, domain);
}

template class GroupNormKernel<float>;