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



#include <ATen/ops/hann_window_ops.h>

namespace at {


// aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor hann_window(int64_t window_length, at::TensorOptions options={}) {
    return at::_ops::hann_window::call(window_length, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
// aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor hann_window(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::hann_window::call(window_length, dtype, layout, device, pin_memory);
}

// aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor hann_window(int64_t window_length, bool periodic, at::TensorOptions options={}) {
    return at::_ops::hann_window_periodic::call(window_length, periodic, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
// aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor hann_window(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::hann_window_periodic::call(window_length, periodic, dtype, layout, device, pin_memory);
}

// aten::hann_window.out(int window_length, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & hann_window_out(at::Tensor & out, int64_t window_length) {
    return at::_ops::hann_window_out::call(window_length, out);
}
// aten::hann_window.out(int window_length, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & hann_window_outf(int64_t window_length, at::Tensor & out) {
    return at::_ops::hann_window_out::call(window_length, out);
}

// aten::hann_window.periodic_out(int window_length, bool periodic, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & hann_window_out(at::Tensor & out, int64_t window_length, bool periodic) {
    return at::_ops::hann_window_periodic_out::call(window_length, periodic, out);
}
// aten::hann_window.periodic_out(int window_length, bool periodic, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & hann_window_outf(int64_t window_length, bool periodic, at::Tensor & out) {
    return at::_ops::hann_window_periodic_out::call(window_length, periodic, out);
}

}
