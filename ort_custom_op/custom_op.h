/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <iostream>
#include "onnxruntime_cxx_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api);

#ifdef __cplusplus
}
#endif


struct Input {
    const char *name;
    std::vector<int64_t> dims;
    std::vector<float> values;
};

struct OrtTensorDimensions : std::vector<int64_t> {
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue *value) {
        OrtTensorTypeAndShapeInfo *info = ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
};

template<typename T>
struct GroupNormKernel {
private:
    float epsilon_;
    Ort::CustomOpApi ort_;

public:
    GroupNormKernel(Ort::CustomOpApi ort, const OrtKernelInfo *info) : ort_(ort) {
        epsilon_ = ort_.KernelInfoGetAttribute<float>(info, "epsilon");
    }

    void Compute(OrtKernelContext *context);
};

struct GroupNormCustomOp : Ort::CustomOpBase<GroupNormCustomOp, GroupNormKernel<float>> {
    void *CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo *info) const {
        return new GroupNormKernel<float>(api, info);
    };

    const char *GetName() const { return "CustomGroupnorm"; };

    size_t GetInputTypeCount() const { return 4; };

    ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

    size_t GetOutputTypeCount() const { return 1; };

    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};