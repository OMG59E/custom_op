#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API eye {
  using schema = at::Tensor (c10::SymInt, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::eye")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "eye(SymInt n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
  static at::Tensor call(c10::SymInt n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, c10::SymInt n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory);
};

struct TORCH_API eye_m {
  using schema = at::Tensor (c10::SymInt, c10::SymInt, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::eye")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "m")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "eye.m(SymInt n, SymInt m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
  static at::Tensor call(c10::SymInt n, c10::SymInt m, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, c10::SymInt n, c10::SymInt m, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory);
};

struct TORCH_API eye_out {
  using schema = at::Tensor & (c10::SymInt, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::eye")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "eye.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)")
  static at::Tensor & call(c10::SymInt n, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, c10::SymInt n, at::Tensor & out);
};

struct TORCH_API eye_m_out {
  using schema = at::Tensor & (c10::SymInt, c10::SymInt, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::eye")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "m_out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "eye.m_out(SymInt n, SymInt m, *, Tensor(a!) out) -> Tensor(a!)")
  static at::Tensor & call(c10::SymInt n, c10::SymInt m, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, c10::SymInt n, c10::SymInt m, at::Tensor & out);
};

}} // namespace at::_ops
