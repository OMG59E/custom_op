#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/scatter_reduce_ops.h>

namespace at {


// aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor
inline at::Tensor scatter_reduce(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self=true) {
    return at::_ops::scatter_reduce_two::call(self, dim, index, src, reduce, include_self);
}

// aten::scatter_reduce.two_out(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & scatter_reduce_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self=true) {
    return at::_ops::scatter_reduce_two_out::call(self, dim, index, src, reduce, include_self, out);
}
// aten::scatter_reduce.two_out(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & scatter_reduce_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self, at::Tensor & out) {
    return at::_ops::scatter_reduce_two_out::call(self, dim, index, src, reduce, include_self, out);
}

}
