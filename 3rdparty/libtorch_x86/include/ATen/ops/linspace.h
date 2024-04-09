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



#include <ATen/ops/linspace_ops.h>

namespace at {


// aten::linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor linspace(const at::Scalar & start, const at::Scalar & end, int64_t steps, at::TensorOptions options={}) {
    return at::_ops::linspace::call(start, end, steps, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
// aten::linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor linspace(const at::Scalar & start, const at::Scalar & end, int64_t steps, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::linspace::call(start, end, steps, dtype, layout, device, pin_memory);
}

// aten::linspace.Tensor_Tensor(Tensor start, Tensor end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor linspace(const at::Tensor & start, const at::Tensor & end, int64_t steps, at::TensorOptions options={}) {
    return at::_ops::linspace_Tensor_Tensor::call(start, end, steps, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
// aten::linspace.Tensor_Tensor(Tensor start, Tensor end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor linspace(const at::Tensor & start, const at::Tensor & end, int64_t steps, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::linspace_Tensor_Tensor::call(start, end, steps, dtype, layout, device, pin_memory);
}

// aten::linspace.Tensor_Scalar(Tensor start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor linspace(const at::Tensor & start, const at::Scalar & end, int64_t steps, at::TensorOptions options={}) {
    return at::_ops::linspace_Tensor_Scalar::call(start, end, steps, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
// aten::linspace.Tensor_Scalar(Tensor start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor linspace(const at::Tensor & start, const at::Scalar & end, int64_t steps, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::linspace_Tensor_Scalar::call(start, end, steps, dtype, layout, device, pin_memory);
}

// aten::linspace.Scalar_Tensor(Scalar start, Tensor end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor linspace(const at::Scalar & start, const at::Tensor & end, int64_t steps, at::TensorOptions options={}) {
    return at::_ops::linspace_Scalar_Tensor::call(start, end, steps, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
// aten::linspace.Scalar_Tensor(Scalar start, Tensor end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor linspace(const at::Scalar & start, const at::Tensor & end, int64_t steps, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::linspace_Scalar_Tensor::call(start, end, steps, dtype, layout, device, pin_memory);
}

// aten::linspace.out(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linspace_out(at::Tensor & out, const at::Scalar & start, const at::Scalar & end, int64_t steps) {
    return at::_ops::linspace_out::call(start, end, steps, out);
}
// aten::linspace.out(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linspace_outf(const at::Scalar & start, const at::Scalar & end, int64_t steps, at::Tensor & out) {
    return at::_ops::linspace_out::call(start, end, steps, out);
}

// aten::linspace.Tensor_Tensor_out(Tensor start, Tensor end, int steps, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linspace_out(at::Tensor & out, const at::Tensor & start, const at::Tensor & end, int64_t steps) {
    return at::_ops::linspace_Tensor_Tensor_out::call(start, end, steps, out);
}
// aten::linspace.Tensor_Tensor_out(Tensor start, Tensor end, int steps, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linspace_outf(const at::Tensor & start, const at::Tensor & end, int64_t steps, at::Tensor & out) {
    return at::_ops::linspace_Tensor_Tensor_out::call(start, end, steps, out);
}

// aten::linspace.Tensor_Scalar_out(Tensor start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linspace_out(at::Tensor & out, const at::Tensor & start, const at::Scalar & end, int64_t steps) {
    return at::_ops::linspace_Tensor_Scalar_out::call(start, end, steps, out);
}
// aten::linspace.Tensor_Scalar_out(Tensor start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linspace_outf(const at::Tensor & start, const at::Scalar & end, int64_t steps, at::Tensor & out) {
    return at::_ops::linspace_Tensor_Scalar_out::call(start, end, steps, out);
}

// aten::linspace.Scalar_Tensor_out(Scalar start, Tensor end, int steps, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linspace_out(at::Tensor & out, const at::Scalar & start, const at::Tensor & end, int64_t steps) {
    return at::_ops::linspace_Scalar_Tensor_out::call(start, end, steps, out);
}
// aten::linspace.Scalar_Tensor_out(Scalar start, Tensor end, int steps, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linspace_outf(const at::Scalar & start, const at::Tensor & end, int64_t steps, at::Tensor & out) {
    return at::_ops::linspace_Scalar_Tensor_out::call(start, end, steps, out);
}

}
