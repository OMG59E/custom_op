# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument, logging-format-interpolation, arguments-differ, invalid-name
# pylint: disable=global-variable-not-assigned, import-outside-toplevel, unnecessary-comprehension
# pylint: disable=unnecessary-list-index-lookup, bare-except, unused-variable
# pylint: disable=broad-exception-caught, eval-used
"""extend onnx op"""
import os
import warnings
import time
import logging
from collections import OrderedDict, Counter
import numpy as np
import onnx

import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.relay.frontend.onnx import (
    OnnxOpConverter,
    _get_convert_map,
    GraphProto,
    _identity_list,
    onnx_input,
    export_model,
    get_source_name,
    get_info,
    get_type,
    dimension_picker,
    dimension_constraint,
    ONNX_DEFAULT_CONFIGS,
    get_pad_pair,
    onnx_default_layout,
    onnx_storage_order2layout,
    get_numpy,
    Conv,
    Pool,
    MatMul,
    Shape,
    Squeeze,
    Unsqueeze,
    Pow,
    Softmax,
    Expand,
    Slice,
    Gemm,
    ReduceL2,
    Reduce,
    Concat,
    Reshape,
    Gather,
    Elemwise,
    GlobalAveragePool,
    GlobalMaxPool,
    Pad,
    Upsample,
    Split,
    ConstantOfShape,
    Not,
    Where,
    Cast,
    Reciprocal,
    BatchNorm,
    InstanceNorm,
    Range,
    ScatterND,
    ScatterElements,
    GatherElements,
    Round,
    Resize,
    LayerNormalization,
    Flatten,
    QuantizeLinear,
    DequantizeLinear,
    QLinearAdd,
    QLinearMul,
    QLinearConv,
    QGemm,
    QLinearConcat,
    QLinearMatMul,
    QLinearSigmoid,
    QLinearSoftmax,
    QLinearAveragePool,
    QLinearGlobalAveragePool,
    QLinearLeakyRelu,
)
from tvm.topi.utils import get_const_tuple

from tvm.relay import op as _op
from tvm.relay import qnn as _qnn
from tvm.relay import expr as _expr
from tvm.relay import ty as _ty
from tvm.relay import function as _function
from tvm.relay import analysis, TupleWrapper
from tvm.relay.frontend.common import (
    infer_type,
    infer_shape,
    get_relay_op,
    fold_constant,
    set_span,
    AttrCvt,
    autopad,
    shape_of,
    infer_channels,
    get_name,
    Renamer,
    try_resolve_var_to_const,
)
from .qir.qir_ops import qir_convert_map
from .register_custom_op_helper import make_custom_op, check_inside_custom_ops_list
from ..qnn.op import edgex_add, edgex_mul, edgex_leaky_relu, edgex_conv3d

logger = logging.getLogger(name="edgex")


class StaticNms(OnnxOpConverter):
    """Operator converter for StaticNms"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if len(inputs) != 7:
            raise ValueError("static_nms expects 7 inputs")
        boxes = inputs[0]
        scores = inputs[1]
        iou_threshold = inputs[2].data.asnumpy().item()
        score_threshold = inputs[3].data.asnumpy().item()
        max_output_size = inputs[4].data.asnumpy().item()
        top_k = inputs[5].data.asnumpy().item()
        invalid_to_bottom = inputs[6].data.asnumpy().item() == 1

        scores = _op.expand_dims(scores, axis=-1, num_newaxis=1)
        data = _op.concatenate([scores, boxes], axis=-1)
        # data = _op.expand_dims(data, axis=0, num_newaxis=1)  # remove batch dim to inputs

        ret = _op.vision.get_valid_counts(
            data, score_threshold=score_threshold, id_index=-1, score_index=0
        )

        nms_out = _op.vision.non_max_suppression(
            ret[1],
            ret[0],
            ret[2],
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            force_suppress=True,  # regardless of class_id
            top_k=top_k,
            coord_start=2,
            score_index=1,
            id_index=0,
            return_indices=False,
            invalid_to_bottom=invalid_to_bottom,
        )

        return nms_out


class StaticBatchedNms(OnnxOpConverter):
    """Operator converter for StaticBatchedNms"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if len(inputs) != 8:
            raise ValueError("static_batched_nms expects 8 inputs")
        boxes = inputs[0]
        scores = inputs[1]
        idxs = inputs[2]
        iou_threshold = inputs[3].data.asnumpy().item()
        score_threshold = inputs[4].data.asnumpy().item()
        max_output_size = inputs[5].data.asnumpy().item()
        top_k = inputs[6].data.asnumpy().item()
        invalid_to_bottom = inputs[7].data.asnumpy().item() == 1

        dtype = infer_type(boxes).checked_type.dtype
        scores = _op.expand_dims(scores, axis=-1, num_newaxis=1)
        idxs = _op.expand_dims(idxs, axis=-1, num_newaxis=1).astype(dtype)
        data = _op.concatenate([idxs, scores, boxes], axis=-1)
        # data = _op.expand_dims(data, axis=0, num_newaxis=1)  # remove batch dim to inputs

        ret = _op.vision.get_valid_counts(
            data, score_threshold=score_threshold, id_index=0, score_index=1
        )

        nms_out = _op.vision.non_max_suppression(
            ret[1],
            ret[0],
            ret[2],
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            force_suppress=False,
            top_k=top_k,
            coord_start=2,
            score_index=1,
            id_index=0,
            return_indices=False,
            invalid_to_bottom=invalid_to_bottom,
        )

        return nms_out


def nonzero_in_it(x):
    assert isinstance(x, (tuple, list)), "x must be a tuple or list!"
    # res = False if 0 in x else True
    res = 0 not in x
    return res


def immediate_compute(op_name, inputs, is_concat=False, is_split=False, is_where=False, **kwargs):
    """immediate_compute"""
    check_inputs = [isinstance(i, _expr.Constant) for i in inputs]
    cond = False
    res = None
    if all(check_inputs):
        cond = True
        const_inputs = [i.data.numpy() for i in inputs]
        out = (
            op_name(*const_inputs, **kwargs)
            if not is_concat
            else op_name(tuple(const_inputs), **kwargs)
        )
        out_dtype = inputs[0].data.dtype if not is_where else inputs[1].data.dtype
        if not is_split:
            res = _expr.const(out, dtype=out_dtype)
        else:
            res = []
            for s in out:
                res.append(_expr.const(s, dtype=out_dtype))

    return cond, res


class Dropout(OnnxOpConverter):
    """Operator converter for dropout."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        out = AttrCvt("dropout", {"ratio": "rate"}, ignores=["is_test"])(inputs, attr, params)
        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)
        return out

    @classmethod
    def _impl_v12(cls, inputs, attr, params):
        """only consider what can be supported by HW"""
        data = inputs[0]
        ratio = inputs[1].data.numpy().item() if len(inputs) > 1 else 0
        if len(inputs) == 3:
            training_mode = inputs[2].data.numpy().item()
            if not training_mode:
                return data
            else:
                if ratio != 0:
                    raise NotImplementedError(
                        "opset version >= 12 of {} not implemented when training_mode is True "
                        "and ratio is {}".format(cls.__name__, ratio)
                    )

        out = _op.nn.dropout(data, ratio)
        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)
        return out


class Clip(OnnxOpConverter):
    """Operator converter for Clip."""

    @staticmethod
    def convert_attributes(inputs, attr, params):
        cond, res = immediate_compute(
            np.clip, inputs, **{"a_min": attr["min"], "a_max": attr["max"]}
        )
        if cond:
            return res
        convert = AttrCvt("clip", transforms={"min": "a_min", "max": "a_max"})
        return convert(inputs, attr, params)

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if "min" not in attr:
            attr["min"] = -np.inf
        if "max" not in attr:
            attr["max"] = np.inf
        return Clip.convert_attributes(inputs, attr, params)

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        inputs_ = inputs
        if len(inputs) == 3 and isinstance(inputs[2], _expr.Constant):
            attr["max"] = inputs[2].data.numpy().item()
            inputs = inputs[0:2]
        if len(inputs) >= 2 and isinstance(inputs[1], _expr.Constant):
            attr["min"] = inputs[1].data.numpy().item()
            inputs = inputs[0:1]
        if "min" in attr and "max" in attr:
            return Clip.convert_attributes(inputs, attr, params)

        inputs = inputs_
        assert len(inputs) <= 3, "Clip-11 takes up to 3 inputs, input, min, max"

        def get_clip_val(data, is_min=True):
            """get_clip_val"""
            if not data:
                clip_val = -np.inf if is_min else np.inf
            else:
                clip_val = None
                if isinstance(data, _expr.Constant):
                    clip_val = data.data.numpy().item()
            return clip_val

        min_val = get_clip_val(inputs[1], is_min=True)
        max_val = get_clip_val(inputs[2], is_min=False)
        if min_val and max_val:
            cond, res = immediate_compute(
                np.clip, [inputs[0]], **{"a_min": min_val, "a_max": max_val}
            )
            if cond:
                return res

        result = inputs[0]
        for i, op in enumerate([_op.tensor.maximum, _op.tensor.minimum]):
            if i < len(inputs) - 1:
                if inputs[i + 1] is not None:
                    result = op(result, inputs[i + 1])
        return result


class ConvOpt(Conv):
    """Operator converter for Conv with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # Use shape of input to determine convolution type.
        data = inputs[0]
        kernel = inputs[1]
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            input_shape = attr["inferred_shape"][0]
        else:
            input_shape = infer_shape(data)
        if "inferred_shape" in attr:
            del attr["inferred_shape"]
        ndim = len(input_shape)

        kernel_type = infer_type(inputs[1])
        kernel_shapes = [get_const_tuple(kernel_type.checked_type.shape)]

        if "kernel_shape" not in attr:
            attr["kernel_shape"] = kernel_shapes[0][2:]

        if "auto_pad" in attr:
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                # Warning: Convolution does not yet support dynamic shapes,
                # one will need to run dynamic_to_static on this model after import
                data = autopad(
                    data,
                    attr.get("strides", [1] * (ndim - 2)),
                    attr["kernel_shape"],
                    attr.get("dilations", [1] * (ndim - 2)),
                    mode=attr["auto_pad"],
                )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = [0 for i in range(ndim - 2)]
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = (
                    f'Value {attr["auto_pad"]} in attribute "auto_pad" of operator Conv '
                    f"is invalid."
                )
                raise tvm.error.OpAttributeInvalid(msg)
            attr.pop("auto_pad")

        attr["channels"] = kernel_shapes[0][0]
        out = AttrCvt(
            op_name=dimension_picker("conv"),
            transforms={
                "kernel_shape": "kernel_size",
                "dilations": ("dilation", 1),
                "pads": ("padding", 0),
                "group": ("groups", 1),
            },
            custom_check=dimension_constraint(),
        )([data, kernel], attr, params)

        use_bias = len(inputs) == 3
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])

        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)

        return out


class PoolOpt(Pool):
    """A helper class for pool op converters with optimization."""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        data = inputs[0]
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            input_shape = attr["inferred_shape"][0]
        else:
            data_infer = infer_type(data)
            input_shape = list(get_const_tuple(data_infer.checked_type.shape))
            if "inferred_shape" not in attr:
                attr["inferred_shape"] = [input_shape]
            else:
                attr["inferred_shape"][0] = input_shape
            dtype = data_infer.checked_type.dtype
            if "inferred_dtype" not in attr:
                attr["inferred_dtype"] = [dtype]
            else:
                attr["inferred_dtype"][0] = dtype
        ndim = len(input_shape)

        attr_cvt, data, compensation_factor = cls._run_calculation(inputs, attr, params)
        if "inferred_shape" in attr:
            del attr["inferred_shape"]
        if "inferred_dtype" in attr:
            input_dtype = attr["inferred_dtype"][0]
            del attr["inferred_dtype"]
        out = attr_cvt([data], attr, params)

        if ndim - len(attr["kernel_shape"]) == 1:
            out = _op.squeeze(out, axis=[0])

        if compensation_factor and "int" not in input_dtype:
            out = _op.multiply(out, _expr.const(compensation_factor, input_dtype))

        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)

        return out

    @classmethod
    def _run_calculation(cls, inputs, attr, params):
        """Helper method to return the processed input data and AttrCvt object"""

        data = inputs[0]

        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][0]
            and nonzero_in_it(attr["inferred_shape"][0])
        ):
            input_shape = attr["inferred_shape"][0]
        else:
            input_shape = infer_shape(data)
        if "inferred_dtype" in attr and attr["inferred_dtype"][0]:
            input_dtype = attr["inferred_dtype"][0]
        else:
            input_dtype = infer_type(data).checked_type.dtype
            if "inferred_shape" not in attr:
                attr["inferred_shape"] = [input_dtype]
            else:
                attr["inferred_shape"][0] = input_dtype

        ndim = len(input_shape)
        if "auto_pad" in attr:
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                if cls.name == "avg_pool":
                    pad_tuple = []
                    for axis in range(len(input_shape) - 2):
                        axis_shape = input_shape[2 + axis]
                        stride = attr.get("strides", [1] * ndim)[axis]
                        kernel = attr["kernel_shape"][axis]
                        pad = get_pad_pair(axis_shape, kernel, stride, attr["auto_pad"])
                        pad_tuple.append(pad)
                    pad_tuple = tuple([val for pair in zip(*pad_tuple) for val in pair])
                    attr["pads"] = pad_tuple
                else:
                    # Warning: Pool does not yet support dynamic shapes,
                    # one will need to run dynamic_to_static on this model after import
                    if "int" in input_dtype:
                        pad_val = np.iinfo(np.dtype(input_dtype)).min
                    else:
                        pad_val = np.finfo(np.dtype(input_dtype)).min
                    data = autopad(
                        data,
                        attr.get("strides", [1] * (ndim - 2)),
                        attr["kernel_shape"],
                        [1] * ndim,
                        pad_value=pad_val,
                        mode=attr["auto_pad"],
                    )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = tuple([0 for i in range(ndim - 2)])
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = (
                    f'Value {attr["auto_pad"]} in attribute "auto_pad" of operator {cls.name} '
                    f"is invalid."
                )
                raise tvm.error.OpAttributeInvalid(msg)
            attr.pop("auto_pad")

        if "storage_order" in attr:
            attr["layout"] = onnx_storage_order2layout(
                attr["storage_order"], dims=(len(input_shape) - 2), op_name=cls.name
            )
        else:
            if ndim - len(attr["kernel_shape"]) == 1:
                data = _op.expand_dims(data, axis=0)
                input_shape = [1] + list(input_shape)

            attr["layout"] = onnx_default_layout(dims=(len(input_shape) - 2), op_name=cls.name)

        # only use for special condition(rare, but condition is complex): efficientnet-b1.onnx
        compensation_factor = None
        if cls.name == "avg_pool":
            kernel_shape = attr["kernel_shape"]
            feature_shape = input_shape[2:]
            assert len(kernel_shape) == len(
                feature_shape
            ), "the dim of kernel_shape should be equal to the dim of feature_shape"
            check_kernel = [size > feature_shape[i] for i, size in enumerate(kernel_shape)]
            if all(check_kernel):
                stride = attr.get("strides", [1] * (ndim - 2))
                pads = list(attr.get("pads", [0] * 2 * (ndim - 2)))
                # onnx framework only use [x1_begin, x2_begin...x1_end, x2_end,...].
                # Maybe modify above
                if len(pads) != 2 * (ndim - 2):
                    pads = [0] * 2 * (ndim - 2)
                offset = [0] * (ndim - 2)
                if attr.get("ceil_mode", False):  # pad after
                    for i in range(ndim - 2):
                        offset[i] = pads[i + ndim - 2] + stride[i] - 1
                out_shape = []
                for i, feature_size in enumerate(feature_shape):
                    out_size = (
                        feature_size - kernel_shape[i] + pads[i] + pads[i + ndim - 2] + offset[i]
                    ) // stride[i] + 1
                    out_shape.append(out_size)

                if out_shape == [1] * (ndim - 2):
                    count_include_pad = attr.get("count_include_pad", False)
                    pad_feature_shape = [
                        feature_size + pads[i] + pads[i + ndim - 2]
                        for i, feature_size in enumerate(feature_shape)
                    ]
                    attr["kernel_shape"] = tuple(pad_feature_shape)
                    attr["strides"] = tuple(pad_feature_shape)
                    if count_include_pad:
                        calc_avg_factor = np.prod(pad_feature_shape).astype("float32")
                    else:
                        calc_avg_factor = np.prod(feature_shape).astype("float32")
                    onnx_avg_factor = np.prod(kernel_shape).astype("float32")
                    compensation_factor = calc_avg_factor / onnx_avg_factor

        return (
            AttrCvt(
                op_name=dimension_picker(cls.name),
                transforms={
                    "kernel_shape": "pool_size",
                    "pads": ("padding", 0),
                    "dilations": ("dilation", 1),
                },
                ignores=["storage_order"],
                custom_check=dimension_constraint(),
            ),
            data,
            compensation_factor,
        )


class AveragePoolOpt(PoolOpt):
    """Operator converter for AveragePool with optimization."""

    name = "avg_pool"


class MaxPoolOpt(PoolOpt):
    """Operator converter for MaxPool with optimization."""

    name = "max_pool"


class BatchNormOpt(BatchNorm):
    """Operator converter for BatchNorm with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # TODO(zhreshold): 'spatial' is not properly handled here.
        # TODO(vvchernov): 'training_mode' (onnx tag) is not correctly handled, ignore for now
        out = AttrCvt(
            op_name="batch_norm",
            ignores=["spatial", "is_test", "consumed_inputs", "momentum", "training_mode"],
        )(inputs, attr, params)
        # We only support test mode, so we return data, moving_mean, moving_var,
        # and then moving_mean and moving_var again as placeholders for
        # the expected "saved_mean", "saved_var".
        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)
        return _expr.TupleWrapper(_expr.Tuple((*out, out[1], out[2])), 5)


class InstanceNormOpt(InstanceNorm):
    """Operator converter for InstanceNorm with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        attr["axis"] = 1
        out = AttrCvt(op_name="instance_norm")(inputs, attr, params)
        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)
        return out


def flatten_to_nd(x, x_shape, nd=3):
    """Flatten input tensor to nd rank"""

    if isinstance(x_shape, _expr.Constant):
        ndims = len(x_shape.data.numpy())
    else:
        ndims = infer_shape(x_shape)[0]

    if ndims == nd:
        return x

    if isinstance(x_shape, _expr.Constant):
        newshape = [-1] + list(x_shape.data.numpy())[ndims - nd + 1 : ndims]
        if isinstance(x, _expr.Constant):
            out = _expr.const(np.reshape(x.data.numpy(), newshape), dtype=x.data.dtype)
        else:
            out = _op.reshape(x, newshape)
    else:
        newshape = _op.concatenate(
            [
                _expr.const([-1], dtype=infer_type(x_shape).checked_type.dtype),
                _op.strided_slice(x_shape, [ndims - nd + 1], [ndims]),
            ],
            0,
        )
        out = _op.reshape(x, fold_constant(newshape))
    return out


def matmul_out_dtype(inputs, out_dtype, inferred_inputs_shape=None):
    """Common function to handle MatMul and MatMulInteger16"""

    if (
        inferred_inputs_shape
        and inferred_inputs_shape[0]
        and nonzero_in_it(inferred_inputs_shape[0])
    ):
        a_shape = _expr.const(inferred_inputs_shape[0], "int64")
        a_rank = len(inferred_inputs_shape[0])
    else:
        a_infer = infer_type(inputs[0])
        if not _ty.is_dynamic(a_infer.checked_type):
            ishape = list(get_const_tuple(a_infer.checked_type.shape))
            a_shape = _expr.const(ishape, "int64")
            a_rank = len(ishape)
        else:
            # a_shape = shape_of(inputs[0])
            # a_rank = infer_shape(a_shape)[0]
            a_shape = _op.shape_of(inputs[0], "int64")
            a_rank = len(a_infer.checked_type.shape)
    b_type = None
    if (
        inferred_inputs_shape
        and inferred_inputs_shape[1]
        and nonzero_in_it(inferred_inputs_shape[1])
    ):
        b_shape = _expr.const(inferred_inputs_shape[1], "int64")
        b_rank = len(inferred_inputs_shape[1])
    else:
        b_infer = infer_type(inputs[1])
        if not _ty.is_dynamic(b_infer.checked_type):
            ishape = list(get_const_tuple(b_infer.checked_type.shape))
            b_shape = _expr.const(ishape, "int64")
            b_rank = len(ishape)
        else:
            # b_shape = shape_of(inputs[1])
            # b_rank = infer_shape(b_shape)[0]
            b_shape = _op.shape_of(inputs[1], "int64")
            b_rank = len(b_infer.checked_type.shape)
        b_type = b_infer

    if (
        isinstance(inputs[0], _expr.Constant)
        and (inputs[0].data.numpy() == 0).all()
        and isinstance(b_shape, _expr.Constant)
    ):
        inputs_dtype = inputs[0].data.dtype
        inputs_1 = _expr.const(
            np.zeros(b_shape.data.numpy(), dtype=inputs_dtype), dtype=inputs_dtype
        )
        cond, res = immediate_compute(np.matmul, [inputs[0], inputs_1], **{})
        if cond:
            return res
    if (
        isinstance(inputs[1], _expr.Constant)
        and (inputs[1].data.numpy() == 0).all()
        and isinstance(a_shape, _expr.Constant)
    ):
        inputs_dtype = inputs[1].data.dtype
        inputs_0 = _expr.const(
            np.zeros(a_shape.data.numpy(), dtype=inputs_dtype), dtype=inputs_dtype
        )
        cond, res = immediate_compute(np.matmul, [inputs_0, inputs[1]], **{})
        if cond:
            return res

    if a_rank > 2 or b_rank > 2:
        # Determine the output batch dimension.
        new_a_shape = a_shape
        new_b_shape = b_shape
        if a_rank > b_rank:
            rank_diff = a_rank - b_rank
            if isinstance(b_shape, _expr.Constant):
                new_b_shape = _expr.const(
                    np.concatenate(
                        [
                            [1] * rank_diff,
                            b_shape.data.numpy(),
                        ],
                        axis=0,
                    ),
                    dtype="int64",
                )
            else:
                new_b_shape = fold_constant(
                    _op.concatenate(
                        [
                            _expr.const(
                                [1] * rank_diff, dtype="int64"
                            ),  # infer_type(b_shape).checked_type.dtype
                            b_shape,
                        ],
                        0,
                    )
                )
        elif a_rank < b_rank:
            rank_diff = b_rank - a_rank
            if isinstance(a_shape, _expr.Constant):
                new_a_shape = _expr.const(
                    np.concatenate(
                        [
                            [1] * rank_diff,
                            a_shape.data.numpy(),
                        ],
                        axis=0,
                    ),
                    dtype="int64",
                )
            else:
                new_a_shape = fold_constant(
                    _op.concatenate(
                        [
                            _expr.const(
                                [1] * rank_diff, dtype="int64"
                            ),  # infer_type(a_shape).checked_type.dtype
                            a_shape,
                        ],
                        0,
                    )
                )
        else:
            pass

        if isinstance(new_a_shape, _expr.Constant) and isinstance(new_b_shape, _expr.Constant):
            batch_end = max(a_rank, b_rank) - 2
            out_batch = _expr.const(
                np.concatenate(
                    [
                        np.maximum(
                            new_b_shape.data.numpy()[0:batch_end],
                            new_a_shape.data.numpy()[0:batch_end],
                        ),
                    ],
                    axis=0,
                ),
                dtype="int64",
            )
        else:
            # out_batch = _op.concatenate(
            #     [
            #         _op.maximum(
            #             _op.strided_slice(new_b_shape, [i], [i + 1]),
            #             _op.strided_slice(new_a_shape, [i], [i + 1]),
            #         )
            #         for i in range(max(a_rank, b_rank) - 2)
            #     ],
            #     0,
            # )
            batch_end = max(a_rank, b_rank) - 2
            out_batch = _op.concatenate(
                [
                    _op.maximum(
                        _op.strided_slice(new_b_shape, [0], [batch_end]),
                        _op.strided_slice(new_a_shape, [0], [batch_end]),
                    )
                ],
                0,
            )

        if not b_type:
            b_type = infer_type(inputs[1])
        # Convert to dense if the second matrix is 2d and non-dynamic
        if b_rank == 2 and not _ty.is_dynamic(b_type.checked_type):
            a = flatten_to_nd(inputs[0], a_shape, 2)
            # if isinstance(inputs[0], _expr.Constant): # move to flatten_to_nd
            #     a = fold_constant(a)
            b = _op.transpose(inputs[1])
            if isinstance(inputs[1], _expr.Constant):
                b = _expr.const(np.transpose(inputs[1].data.numpy()), dtype=inputs[1].data.dtype)
            output = _op.nn.dense(a, b, out_dtype=out_dtype)
        else:
            a = inputs[0]
            b = inputs[1]
            # broadcast a and b
            if isinstance(out_batch, _expr.Constant) and isinstance(a_shape, _expr.Constant):
                a_broadcasted_shape = _expr.const(
                    np.concatenate(
                        [out_batch.data.numpy(), a_shape.data.numpy()[a_rank - 2 : a_rank]], axis=0
                    ),
                    dtype="int64",
                )
            else:
                a_broadcasted_shape = fold_constant(
                    _op.concatenate(
                        [
                            out_batch,
                            fold_constant(_op.strided_slice(a_shape, [a_rank - 2], [a_rank])),
                        ],
                        0,
                    )
                )
            if isinstance(out_batch, _expr.Constant) and isinstance(b_shape, _expr.Constant):
                b_broadcasted_shape = _expr.const(
                    np.concatenate(
                        [out_batch.data.numpy(), b_shape.data.numpy()[b_rank - 2 : b_rank]], axis=0
                    ),
                    dtype="int64",
                )
            else:
                b_broadcasted_shape = fold_constant(
                    _op.concatenate(
                        [
                            out_batch,
                            fold_constant(_op.strided_slice(b_shape, [b_rank - 2], [b_rank])),
                        ],
                        0,
                    )
                )
            if not tvm.ir.structural_equal(a_shape, a_broadcasted_shape):
                if isinstance(a, _expr.Constant) and isinstance(
                    a_broadcasted_shape, _expr.Constant
                ):
                    a = _expr.const(
                        np.broadcast_to(a.data.numpy(), tuple(a_broadcasted_shape.data.numpy())),
                        dtype=a.data.dtype,
                    )
                else:
                    a = _op.transform.broadcast_to(a, a_broadcasted_shape)
            if not tvm.ir.structural_equal(b_shape, b_broadcasted_shape):
                if isinstance(b, _expr.Constant) and isinstance(
                    b_broadcasted_shape, _expr.Constant
                ):
                    b = _expr.const(
                        np.broadcast_to(b.data.numpy(), tuple(b_broadcasted_shape.data.numpy())),
                        dtype=b.data.dtype,
                    )
                else:
                    b = _op.transform.broadcast_to(b, b_broadcasted_shape)
            # Convert a and b into 3 dimensional tensors.
            # a = flatten_to_nd(a, shape_of(a), 3)
            # b = flatten_to_nd(b, shape_of(b), 3)
            a = flatten_to_nd(a, a_broadcasted_shape, 3)
            b = flatten_to_nd(b, b_broadcasted_shape, 3)
            if ONNX_DEFAULT_CONFIGS["use_nt_batch_matmul"]:
                # Transpose matrix dimensions of b.
                if isinstance(b, _expr.Constant):
                    bt = _expr.const(np.transpose(b.data.numpy(), (0, 2, 1)), dtype=b.data.dtype)
                else:
                    bt = _op.transpose(b, [0, 2, 1])
                # Perform a NT batch matmul.
                output = _op.nn.batch_matmul(a, bt, out_dtype=out_dtype)
            else:
                # Perform a NN batch matmul.
                output = _op.nn.batch_matmul(a, b, out_dtype=out_dtype, transpose_b=False)
        # Reshape output to original dimensions.
        if (
            isinstance(out_batch, _expr.Constant)
            and isinstance(a_shape, _expr.Constant)
            and isinstance(b_shape, _expr.Constant)
        ):
            final_shape = _expr.const(
                np.concatenate(
                    [
                        out_batch.data.numpy(),
                        a_shape.data.numpy()[a_rank - 2 : a_rank - 1],
                        b_shape.data.numpy()[b_rank - 1 : b_rank],
                    ],
                    axis=0,
                ),
                dtype="int64",
            )
        else:
            final_shape = _op.concatenate(
                [
                    out_batch,
                    fold_constant(_op.strided_slice(a_shape, [a_rank - 2], [a_rank - 1])),
                    fold_constant(_op.strided_slice(b_shape, [b_rank - 1], [b_rank])),
                ],
                0,
            )
            final_shape = fold_constant(final_shape)

        res = _op.reshape(output, final_shape)
        return res

    if a_rank == 1 or b_rank == 1:
        axis = []
        if a_rank == 1:
            lhs = _op.expand_dims(inputs[0], axis=0)
            axis.append(0)
        else:
            lhs = inputs[0]
        if b_rank == 1:
            rhs = _op.expand_dims(inputs[1], axis=1)
            axis.append(-1)
        else:
            rhs = inputs[1]

        if isinstance(inputs[0], _expr.Constant):
            lhs = fold_constant(lhs)
        if isinstance(inputs[1], _expr.Constant):
            rhs = fold_constant(rhs)
        return _op.squeeze(_op.nn.matmul(lhs, rhs), axis=axis)

    # Otherwise a simple dense op will get the job done.
    input_1_t = _op.transpose(inputs[1], axes=(1, 0))
    if isinstance(inputs[1], _expr.Constant):
        input_1_t = _expr.const(
            np.transpose(inputs[1].data.numpy(), axes=(1, 0)), dtype=inputs[1].data.dtype
        )

    return _op.nn.dense(inputs[0], input_1_t, out_dtype=out_dtype)


class MatMulOpt(MatMul):
    """Operator converter for MatMul with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, f"MatMul op take 2 inputs, {len(inputs)} given"
        # Need to check input shape as batch matmul must be supported.
        cond, res = immediate_compute(np.matmul, inputs, **{})
        if cond:
            return res
        inferred_inputs_shape = attr["inferred_shape"] if "inferred_shape" in attr else None
        if "inferred_dtype" in attr and attr["inferred_dtype"][0]:
            out_dtype = attr["inferred_dtype"][0]
        else:
            input0_infer = infer_type(inputs[0])
            out_dtype = input0_infer.checked_type.dtype
            if not _ty.is_dynamic(input0_infer.checked_type):
                input0_shape = list(get_const_tuple(input0_infer.checked_type.shape))
                if not inferred_inputs_shape:  # --> [] or None
                    inferred_inputs_shape = [input0_shape, None]
                else:  # --> [x, x]
                    inferred_inputs_shape[0] = input0_shape

        return matmul_out_dtype(inputs, out_dtype, inferred_inputs_shape)


class ShapeOpt(Shape):
    """Operator converter for Shape with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][0]
            and nonzero_in_it(attr["inferred_shape"][0])
        ):
            shape_out = _expr.const(attr["inferred_shape"][0], "int64")
        else:
            shape_out = shape_of(inputs[0], "int64")
        return shape_out

    @classmethod
    def _impl_v15(cls, inputs, attr, params):
        start = attr.get("start")
        end = attr.get("end")
        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][0]
            and nonzero_in_it(attr["inferred_shape"][0])
        ):
            input0 = attr["inferred_shape"][0]
            start = start or 0  # default to first
            end = end or len(input0)  # default to last
            shape_out = _expr.const(input0[start:end], "int64")
        else:
            shape_out = shape_of(inputs[0], dtype="int64", start=start, end=end)
        return shape_out


class SqueezeOpt(Squeeze):
    """Operator converter for Squeeze with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axes = attr.get("axes", None)
        if isinstance(axes, list):
            axes = tuple(axes)
        cond, res = immediate_compute(np.squeeze, inputs, **{"axis": axes})
        if cond:
            return res
        return Squeeze._impl_v1(inputs, attr, params)

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][0]
            and nonzero_in_it(attr["inferred_shape"][0])
        ):
            ishape = attr["inferred_shape"][0]
        else:
            ishape = infer_shape(inputs[0])
        axis = inputs[1]

        if axis is None:
            # If axes is not provided, all the single dimensions will be removed from the shape.
            if not ishape:  # scalar
                return inputs[0]

            axis = [i for i in range(len(ishape)) if ishape[i] == 1]
            axis = _op.const(axis)

        dtype = infer_type(axis).checked_type.dtype

        if isinstance(axis, _expr.Constant):
            constant_axes = list(axis.data.numpy())
            constant_axes = list(map(int, constant_axes))
            cond, res = immediate_compute(np.squeeze, [inputs[0]], **{"axis": tuple(constant_axes)})
            if cond:
                return res
            return _op.squeeze(inputs[0], constant_axes)

        rank = _op.shape_of(_op.shape_of(inputs[0], dtype), dtype)
        axis = _op.where(axis < _op.const(0, dtype), axis + rank, axis)
        return _op.squeeze(inputs[0], fold_constant(axis))


class UnsqueezeOpt(Unsqueeze):
    """Operator converter for Unsqueeze with optimization."""

    @classmethod
    def run_calculation(cls, tensor, axes):
        axes = sorted(axes)
        cond, res = immediate_compute(np.expand_dims, [tensor], **{"axis": tuple(axes)})
        if cond:
            return res
        for axis in axes:
            if axis < 0 and isinstance(tensor, _expr.Var):
                axis = len(tensor.type_annotation.concrete_shape) + len(axes) + axis
            tensor = _op.expand_dims(tensor, axis=axis, num_newaxis=1)
        return tensor

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return cls.run_calculation(inputs[0], attr["axes"])

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        if isinstance(inputs[1], _expr.Constant):
            constant_axes = list(inputs[1].data.numpy())
            constant_axes = list(map(int, constant_axes))
            return cls.run_calculation(inputs[0], constant_axes)

        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            ishape = attr["inferred_shape"][0]
            rank_input = len(ishape)
        else:
            rank_input = len(infer_type(inputs[0]).checked_type.shape)
        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][1]
            and nonzero_in_it(attr["inferred_shape"][1])
        ):
            ishape = attr["inferred_shape"][1]
            num_new_axis = int(ishape[0])
        else:
            num_new_axis = int(infer_type(inputs[1]).checked_type.shape[0])
        axes = relay.sort(inputs[1])
        axes = relay.split(axes, num_new_axis).astuple()
        rank_output = rank_input + num_new_axis
        result = inputs[0]

        # TODO (AndrewZhaoLuo): investigate performance issues with consecutive
        # dynamic expand_dims on non-llvm targets.
        for i in range(num_new_axis):
            axis = relay.TupleGetItem(axes, i)
            # Unpack scalar
            axis = relay.reshape(axis, [])
            axis = relay.where(
                axis >= relay.const(0, "int64"), axis, axis + relay.const(rank_output, "int64")
            )
            result = _op.expand_dims(result, axis)
        return result


class PowOpt(Pow):
    """Operator converter for Pow with optimization."""

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        cond, res = immediate_compute(np.power, inputs, **{})
        if cond:
            return res

        x = inputs[0]
        y = inputs[1]

        if "inferred_dtype" in attr and attr["inferred_dtype"][0]:
            x_type = attr["inferred_dtype"][0]
        else:
            x_type = infer_type(x).checked_type.dtype
        output_type = x_type
        y_type = infer_type(y).checked_type.dtype

        if not x_type.startswith("float"):
            x_type = "float32"
            x = _op.cast(x, x_type)

        if x_type != y_type:
            y = _op.cast(y, x_type)

        # TODO: come up with good default integer pow() func for common backends
        result = _op.power(x, y)
        if x_type != output_type:
            return _op.cast(result, output_type)
        return result


class SoftmaxOpt(Softmax):
    """Operator converter for Softmax with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get("axis", 1)
        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][0]
            and nonzero_in_it(attr["inferred_shape"][0])
        ):
            in_shape = attr["inferred_shape"][0]
        else:
            in_shape = infer_shape(inputs[0])
        ndim = len(in_shape)
        if axis < 0:
            axis += ndim
        if axis == 0:
            reshape_shape = [-1]
        elif axis == ndim - 1:
            out = _op.nn.softmax(inputs[0], axis=axis)
            if isinstance(inputs[0], _expr.Constant):
                out = fold_constant(out)
            return out
        else:
            axis_val = [in_shape[i] for i in range(axis)]
            reshape_shape = [np.prod(axis_val)] + [-1]
        data_reshape = _op.reshape(inputs[0], newshape=reshape_shape)
        out = _op.nn.softmax(data_reshape, axis=-1)
        out = _op.reshape(out, newshape=in_shape)
        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)

        return out

    @classmethod
    def _impl_v13(cls, inputs, attr, _):
        axis = attr.get("axis", -1)
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            ndim = len(attr["inferred_shape"][0])
        else:
            ndim = len(infer_shape(inputs[0]))
        if axis < 0:
            axis += ndim
        out = _op.nn.softmax(inputs[0], axis=axis)
        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)

        return out


class ExpandOpt(Expand):
    """Operator converter for Expand with optimization."""

    @classmethod
    def _impl_v8(cls, inputs, attr, params):
        dtype = infer_type(inputs[1]).checked_type.dtype
        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][0]
            and nonzero_in_it(attr["inferred_shape"][0])
        ):
            in_shape = _expr.const(attr["inferred_shape"][0], dtype)
        else:
            in_shape = shape_of(inputs[0], dtype=dtype)
        shape = inputs[1]

        # Currently 'op.broadcast_to' expect the rank of the given 'shape'
        # (the 2nd input) is always higher than that of the given 'input' (the 1st input)
        # However, ONNX Expand supports multi-directional broadcasting, which allows
        # above pattern and also some extent of 'shape' can be smaller than the corresponding
        # extent of 'input'. In this case, the extent of 'shape' must be 1.
        # https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
        # In above cases, we cannot directorly apply 'op.broadcast_to' instead of 'expand'
        # so, here we solved this problem by expanding the given 'shape' itself.

        def expand_shape(in_shape, shape):
            """A function expands the shape when the rank is lower than that of the given
            intput. Also it replaces the extent of the shape with the corresponding extent
            of the intput when it is 1.
            """
            if isinstance(in_shape, _expr.Constant):
                in_dims = len(in_shape.data.numpy())
            else:
                in_dims = infer_shape(in_shape)[0]
            if isinstance(shape, _expr.Constant):
                new_dims = len(shape.data.numpy())
            else:
                new_dims = infer_shape(shape)[0]

            if in_dims < new_dims:
                diff_dims = new_dims - in_dims
                if isinstance(in_shape, _expr.Constant):
                    in_shape = _expr.const(
                        np.concatenate([[1] * diff_dims, in_shape.data.numpy()], axis=0),
                        dtype=dtype,
                    )
                else:
                    in_shape = fold_constant(
                        _op.concatenate(
                            [_expr.const([1] * diff_dims, dtype=dtype), in_shape], axis=0
                        )
                    )
            elif new_dims < in_dims:
                diff_dims = in_dims - new_dims
                if isinstance(shape, _expr.Constant):
                    shape = _expr.const(
                        np.concatenate([[1] * diff_dims, shape.data.numpy()], axis=0), dtype=dtype
                    )
                else:
                    shape = fold_constant(
                        _op.concatenate([_expr.const([1] * diff_dims, dtype=dtype), shape], axis=0)
                    )

            if isinstance(in_shape, _expr.Constant) and isinstance(shape, _expr.Constant):
                res = np.maximum(in_shape.data.numpy(), shape.data.numpy())
                new_shape = _expr.const(res, in_shape.data.dtype)
            else:
                new_shape = fold_constant(_op.maximum(in_shape, shape))

            return new_shape

        shape = expand_shape(in_shape, shape)
        if isinstance(inputs[0], _expr.Constant) and isinstance(shape, _expr.Constant):
            return _expr.const(
                np.broadcast_to(inputs[0].data.numpy(), tuple(shape.data.numpy())),
                dtype=inputs[0].data.dtype,
            )
        return _op.broadcast_to(inputs[0], shape=shape)


class SliceOpt(Slice):
    """Operator converter for Slice with optimization."""

    @classmethod
    def _immediate_compute(cls, x, begin, end, steps=None, axes=None):
        if steps is None:
            steps = list(np.ones_like(np.asarray(begin)))
        if axes is None:
            axes = list(np.arange(len(begin)))

        data_np = x.data.numpy()
        dtype = x.data.dtype
        axes_max = np.max(np.asarray(axes)) + 1
        slice_list = [slice(None, None, None)] * axes_max
        for i, axis in enumerate(axes):
            slice_list[axis] = slice(begin[i], end[i], steps[i])

        return _expr.const(data_np[tuple(slice_list)], dtype=dtype)

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if isinstance(attr["starts"], int):
            attr["starts"] = (attr["starts"],)
            attr["ends"] = (attr["ends"],)

        try:
            # Update the starts and ends according to axes if required.
            if isinstance(attr["axes"], int):
                attr["axes"] = (attr["axes"],)
            new_starts, new_ends, new_axes = cls._common(attr["starts"], attr["ends"], attr["axes"])
            attr["axes"] = new_axes
            attr["starts"] = new_starts
            attr["ends"] = new_ends
        except KeyError:
            pass
        begin = list(attr["starts"])
        end = list(attr["ends"])

        if isinstance(inputs[0], _expr.Constant):
            return cls._immediate_compute(inputs[0], begin, end)
        return _op.strided_slice(inputs[0], begin=begin, end=end)

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        starts = inputs[1]
        ends = inputs[2]
        axes = inputs[3]
        steps = inputs[4]

        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            ishape = attr["inferred_shape"][0]
        else:
            # ishape = infer_shape(inputs[0])
            input0_infer = infer_type(inputs[0])
            ishape = list(get_const_tuple(input0_infer.checked_type.shape))
            if not _ty.is_dynamic(input0_infer.checked_type):
                if "inferred_shape" not in attr:
                    attr["inferred_shape"] = [ishape]
                else:
                    attr["inferred_shape"][0] = ishape
        data_rank = len(ishape)

        if axes is not None:
            # Normalize for negative axes
            if isinstance(axes, _expr.Constant):
                axes_dtype = axes.data.dtype
                axes_np = axes.data.numpy()
                res = np.where(axes_np < 0, axes_np + data_rank, axes_np)
                axes = _expr.const(res, axes_dtype)
            else:
                axes_dtype = infer_type(axes).checked_type.dtype
                axes = fold_constant(
                    _op.where(
                        axes < _op.const(0, axes_dtype),
                        axes + _op.const(data_rank, axes_dtype),
                        axes,
                    )
                )

        def has_static_axes():
            return (
                isinstance(starts, _expr.Constant)
                and isinstance(ends, _expr.Constant)
                and (steps is None or isinstance(steps, _expr.Constant))
            )

        if has_static_axes():
            begin_np = starts.data.numpy().astype("int64")
            end_np = ends.data.numpy().astype("int64")
            if steps is None:
                strides_np = np.ones_like(begin_np).astype("int64")
            else:
                strides_np = steps.data.numpy().astype("int64")
            begin_np = list(begin_np)
            end_np = list(end_np)
            strides_np = list(strides_np)
            if axes is not None and isinstance(axes, _expr.Constant):
                axes_np = axes.data.numpy().astype("int64")
                if all([isinstance(ishape[i], int) for i in axes_np]):
                    if isinstance(inputs[0], _expr.Constant):
                        return cls._immediate_compute(
                            inputs[0],
                            begin_np,
                            end_np,
                            steps=strides_np,
                            axes=list(axes_np),
                        )
                    return _op.strided_slice(
                        inputs[0], begin_np, end_np, strides_np, axes=list(axes_np)
                    )
            if axes is None and isinstance(inputs[0], _expr.Constant):
                return cls._immediate_compute(inputs[0], begin_np, end_np, steps=strides_np)

        # Update the starts and ends according to axes if required.
        if axes is not None:
            if (
                "inferred_shape" in attr
                and attr["inferred_shape"][0]
                and nonzero_in_it(attr["inferred_shape"][0])
            ):
                data_shape = _expr.const(
                    attr["inferred_shape"][0], dtype=infer_type(ends).checked_type.dtype
                )
            else:
                data_shape = _op.shape_of(inputs[0], dtype=infer_type(ends).checked_type.dtype)
            starts = _op.scatter_elements(
                _op.const([0] * data_rank, dtype=infer_type(starts).checked_type.dtype),
                axes,
                starts,
                axis=0,
            )
            ends = _op.scatter_elements(data_shape, axes, ends, axis=0)
            if steps is not None:
                steps = _op.scatter_elements(
                    _op.const([1] * data_rank, dtype=infer_type(steps).checked_type.dtype),
                    axes,
                    steps,
                    axis=0,
                )

        if steps is None:
            steps = _op.const([1] * data_rank, dtype=infer_type(starts).checked_type.dtype)

        return _op.strided_slice(
            inputs[0], fold_constant(starts), fold_constant(ends), fold_constant(steps)
        )


class GemmOpt(Gemm):
    """Operator converter for Gemm with optimization."""

    @classmethod
    def _immediate_compute(cls, A, B, C, transA, transB, alpha, beta, dtype):
        A = A.data.numpy() if transA == 0 else np.transpose(A.data.numpy(), axes=(1, 0))
        B = B.data.numpy() if transB == 0 else np.transpose(B.data.numpy(), axes=(1, 0))
        C = C.data.numpy() if C is not None else np.array(0)

        Y = alpha * np.dot(A, B) + beta * C
        out = _expr.const(Y, dtype=dtype)

        return out

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert (
            len(inputs) == 3 or len(inputs) == 2
        ), f"Gemm op take 2 or 3 inputs, {len(inputs)} given"
        if (
            "inferred_dtype" in attr
            and attr["inferred_dtype"][0]
            and "inferred_shape" in attr
            and attr["inferred_shape"][0]
        ):
            dtype = attr["inferred_dtype"][0]
            ndim = len(attr["inferred_shape"][0])
        else:
            input0_state = infer_type(inputs[0])
            dtype = input0_state.checked_type.dtype
            ndim = len(input0_state.checked_type.shape)
        # Y = alpha * A * B + beta * C
        alpha = float(attr.get("alpha", 1.0))
        beta = float(attr.get("beta", 1.0))
        transA = int(attr.get("transA", 0))
        transB = int(attr.get("transB", 0))
        C = inputs[2] if len(inputs) == 3 else None
        C_cond = False
        if (not C) or (C and isinstance(C, _expr.Constant)):
            C_cond = True
        if (
            isinstance(inputs[0], _expr.Constant)
            and isinstance(inputs[1], _expr.Constant)
            and C_cond
        ):
            return cls._immediate_compute(
                inputs[0], inputs[1], C, transA, transB, alpha, beta, dtype
            )

        # get number of channels
        channels = infer_channels(inputs[1], not transB)
        if transA:
            if isinstance(inputs[0], _expr.Constant):
                inputs[0] = _expr.const(np.transpose(inputs[0].data.numpy(), axes=(1, 0)), dtype)
            else:
                inputs[0] = _op.transpose(inputs[0], axes=(1, 0))
        if not transB:
            if isinstance(inputs[1], _expr.Constant):
                inputs[1] = _expr.const(np.transpose(inputs[1].data.numpy(), axes=(1, 0)), dtype)
            else:
                inputs[1] = _op.transpose(inputs[1], axes=(1, 0))

        if ndim != 2:
            inputs[0] = _op.nn.batch_flatten(inputs[0])
        if alpha != 1.0:
            inputs[0] *= _expr.const(alpha, dtype=dtype)
        out = _op.nn.dense(inputs[0], inputs[1], units=channels)
        if len(inputs) == 3:
            if beta != 1.0:
                # out += _expr.const(float(beta), dtype=dtype) * inputs[2]
                out += _expr.const(float(beta) * inputs[2].data.numpy(), dtype=dtype)
            else:
                out += inputs[2]
        return out


class ReduceL2Opt(ReduceL2):
    """Operator converter for ReduceL2 with optimization."""

    @classmethod
    def _immediate_compute(cls, x, axis=None, keepdims=True):
        data_np = x.data.numpy()
        dtype = x.data.dtype
        res = np.sqrt(np.sum(a=np.square(data_np), axis=axis, keepdims=keepdims))

        return _expr.const(res, dtype=dtype)

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            ishape = attr["inferred_shape"][0]
        else:
            ishape = infer_shape(inputs[0])
        if not ishape:  # promote scalar to 1-D tensor
            inputs[0] = _op.expand_dims(inputs[0], axis=0)
            ishape = [1]

        if "axes" in attr:
            axis = attr.get("axes", 0)
        else:
            axis_len = len(ishape)  # len(infer_shape(inputs[0]))
            axis = list(range(axis_len))
        attr = {"axis": axis, "keepdims": attr.get("keepdims", True)}
        if isinstance(inputs[0], _expr.Constant):
            return cls._immediate_compute(inputs[0], **attr)
        inputs[0] = inputs[0] * inputs[0]
        out = AttrCvt("sum")(inputs, attr)

        return _op.sqrt(out)


class ReduceOpt(Reduce):
    """Operator converter for reduce ops with optimization."""

    name = ""
    func = ""

    @classmethod
    def run_calculation(cls, inputs, axis, keepdims):
        attr = {"axis": axis, "keepdims": keepdims}
        if cls.func:
            if isinstance(axis, list):
                axis = tuple(axis)
            keepdims = bool(keepdims)
            cond, res = immediate_compute(cls.func, inputs, **{"axis": axis, "keepdims": keepdims})
            if cond:
                return res
        return AttrCvt(cls.name)(inputs, attr)

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            ishape = attr["inferred_shape"][0]
        else:
            ishape = infer_shape(inputs[0])
        if not ishape:  # promote scalar to 1-D tensor
            inputs[0] = _op.expand_dims(inputs[0], axis=0)
            ishape = (1,)

        if "axes" in attr:
            axis = attr.get("axes", 0)
        else:
            axis_len = len(ishape)  # len(infer_shape(inputs[0]))
            axis = list(range(axis_len))

        return cls.run_calculation(inputs, axis, attr.get("keepdims", True))

    @classmethod
    def _impl_v12(cls, inputs, attr, params):
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            ishape = attr["inferred_shape"][0]
        else:
            # ishape = infer_shape(inputs[0])
            input0_infer = infer_type(inputs[0])
            if not _ty.is_dynamic(input0_infer.checked_type):
                ishape = list(get_const_tuple(input0_infer.checked_type.shape))
                if "inferred_shape" not in attr:
                    attr["inferred_shape"] = [ishape]
                else:
                    attr["inferred_shape"][0] = ishape
            else:
                ishape = input0_infer.checked_type
        if not ishape:  # promote scalar to 1-D tensor
            inputs[0] = _op.expand_dims(inputs[0], axis=0)
            if "inferred_shape" not in attr:
                attr["inferred_shape"] = [[1]]
            else:
                attr["inferred_shape"][0] = [1]

        if len(inputs) == 2:
            if isinstance(inputs[1], _expr.Constant):
                # Get axis and unpack scalar
                constant_axis = int(inputs[1].data.numpy()[0])
                return cls.run_calculation([inputs[0]], constant_axis, attr.get("keepdims", True))

            raise ValueError("Dynamic Reduce is not supported yet!")

        return cls._impl_v1(inputs, attr, params)

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            ishape = attr["inferred_shape"][0]
        else:
            ishape = infer_shape(inputs[0])
            input0_infer = infer_type(inputs[0])
            if not _ty.is_dynamic(input0_infer.checked_type):
                ishape = list(get_const_tuple(input0_infer.checked_type.shape))
                if "inferred_shape" not in attr:
                    attr["inferred_shape"] = [ishape]
                else:
                    attr["inferred_shape"][0] = ishape
            else:
                ishape = input0_infer.checked_type
        if not ishape:  # promote scalar to 1-D tensor
            inputs[0] = _op.expand_dims(inputs[0], axis=0)
            ishape = [1]
            if "inferred_shape" not in attr:
                attr["inferred_shape"] = [ishape]
            else:
                attr["inferred_shape"][0] = ishape

        noop_with_empty_axes = attr.get("noop_with_empty_axes", 0)
        num_axis = int(infer_type(inputs[1]).checked_type.shape[0]) if inputs[1] is not None else 0

        if noop_with_empty_axes and num_axis == 0:
            return inputs[0]

        if len(inputs) == 2:
            if isinstance(inputs[1], _expr.Constant):
                # Get axis and unpack scalar
                constant_axis = int(inputs[1].data.numpy()[0])
                return cls.run_calculation([inputs[0]], constant_axis, attr.get("keepdims", True))

            if num_axis > 0:
                raise ValueError("Dynamic Reduce is not supported yet!")

            axis_len = len(ishape)  # len(infer_shape(inputs[0]))
            axis = list(range(axis_len))
            return cls.run_calculation([inputs[0]], axis, attr.get("keepdims", True))

        return cls._impl_v1(inputs, attr, params)


class ReduceMaxOpt(ReduceOpt):
    """Operator converter for ReduceMax with optimization."""

    name = "max"
    func = np.maximum.reduce


class ReduceMinOpt(ReduceOpt):
    """Operator converter for ReduceMin with optimization."""

    name = "min"
    func = np.minimum.reduce


class ReduceSumOpt(ReduceOpt):
    """Operator converter for ReduceSum with optimization."""

    name = "sum"
    func = np.sum


class ReduceMeanOpt(ReduceOpt):
    """Operator converter for ReduceMean with optimization."""

    name = "mean"
    func = np.mean


class ReduceProdOpt(ReduceOpt):
    """Operator converter for ReduceProd with optimization."""

    name = "prod"
    func = np.prod


class ReduceLogSumExpOpt(ReduceOpt):
    """Operator converter for ReduceLogSumExp with optimization."""

    name = "logsumexp"
    func = None


class ConcatOpt(Concat):
    """Operator converter for Concat with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        new_inputs = []
        for inp in inputs:
            if get_name(inp) in params:
                new_inputs.append(_expr.const(params[inp.name_hint]))
            else:
                new_inputs.append(inp)
        cond, res = immediate_compute(
            np.concatenate, new_inputs, is_concat=True, **{"axis": attr["axis"]}
        )
        if cond:
            return res

        return AttrCvt(op_name="concatenate")((new_inputs,), attr)


class ReshapeOpt(Reshape):
    """Operator converter for Reshape with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        cond, res = immediate_compute(np.reshape, [inputs[0]], **{"newshape": attr["shape"]})
        if cond:
            return res
        return _op.reshape(inputs[0], attr["shape"])

    @classmethod
    def _impl_v5(cls, inputs, attr, params):
        allowzero = attr.get("allowzero", False)
        if get_name(inputs[1]) in params:
            shape = tuple(params[inputs[1].name_hint].numpy().astype("int32"))
        else:
            shape = inputs[1]
            if isinstance(shape, _expr.Constant):
                shape = tuple(shape.data.numpy())
        if isinstance(shape, tuple):
            if allowzero:
                warnings.warn(f"Unsupport(const compute): allowzero={allowzero}, newshape={shape}")
            new_shape = np.copy(shape)
            if allowzero == 0:
                zeros_index = np.where(shape == 0)
                new_shape[zeros_index] = np.array(shape)[zeros_index]
            cond, res = immediate_compute(np.reshape, [inputs[0]], **{"newshape": new_shape})
            if cond:
                return res
        out = _op.reshape(inputs[0], shape, allowzero=allowzero)
        return out


class TransposeOpt(OnnxOpConverter):
    """Operator converter for Transpose with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        cond, res = immediate_compute(np.transpose, [inputs[0]], **{"axes": attr["perm"]})
        if cond:
            return res
        return AttrCvt("transpose", {"perm": "axes"})(inputs, attr, params)


class RenamerCompute(OnnxOpConverter):
    """Operator converter for RenamerCompute with optimization."""

    name = ""
    func = ""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        cls_name = cls.name
        if cls.name == "leaky_relu":
            alpha = attr.get("alpha", 0.01)
            kwargs = {"alpha": alpha}
            if alpha == 0:
                cls_name = "relu"
                if "alpha" in attr:
                    del attr["alpha"]
                kwargs = {}
        else:
            kwargs = {}

        cond, res = immediate_compute(cls.func, inputs, **kwargs)
        if cond:
            return res
        return Renamer(cls_name)(inputs, attr, params)


def relu_np(x):
    """np compute for relu"""
    return np.maximum(x, 0)


class ReluOpt(RenamerCompute):
    """Operator converter for Relu with optimization."""

    name = "relu"
    func = relu_np


def leaky_relu_np(x, alpha=0.01):
    """np compute for leaky_relu"""
    return np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * alpha


class LeakyReluOpt(RenamerCompute):
    """Operator converter for LeakyRelu with optimization."""

    name = "leaky_relu"
    func = leaky_relu_np


class SqrtOpt(RenamerCompute):
    """Operator converter for Sqrt with optimization."""

    name = "sqrt"
    func = np.sqrt


class ExpOpt(RenamerCompute):
    """Operator converter for Sqrt with optimization."""

    name = "exp"
    func = np.exp


class LogOpt(RenamerCompute):
    """Operator converter for Log with optimization."""

    name = "log"
    func = np.log


def sigmoid_np(x):
    """np compute for sigmoid"""
    return 1.0 / (1.0 + np.exp(np.negative(x)))


class SigmoidOpt(RenamerCompute):
    """Operator converter for Sqrt with optimization."""

    name = "sigmoid"
    func = sigmoid_np


class SinOpt(RenamerCompute):
    """Operator converter for Sin with optimization."""

    name = "sin"
    func = np.sin


class CosOpt(RenamerCompute):
    """Operator converter for Cos with optimization."""

    name = "cos"
    func = np.cos


def normalize_gather_indices(data, indices, axis, attr):
    """Make sure gather indices aren't negative"""
    axis_size = None
    if (
        "inferred_shape" in attr
        and attr["inferred_shape"][0]
        and nonzero_in_it(attr["inferred_shape"][0])
    ):
        data_shape = attr["inferred_shape"][0]
        axis_size = data_shape[axis]
    else:
        data_infer = infer_type(data)
        if not _ty.is_dynamic(data_infer.checked_type):
            data_shape = list(get_const_tuple(data_infer.checked_type.shape))
            axis_size = data_shape[axis]
    if isinstance(indices, _expr.Constant) and axis_size:
        indices_np = indices.data.numpy()
        dtype = indices.data.dtype
        res = np.where(indices_np < 0, indices_np + axis_size, indices_np)
        indices = _expr.const(res, dtype)
        return indices
    if "inferred_dtype" in attr and attr["inferred_dtype"][1]:
        ind_dtype = attr["inferred_dtype"][1]
    else:
        ind_dtype = infer_type(indices).checked_type.dtype
    # Normalize the indices to a positive range
    if axis_size:  # data_shape can be obtain, ->data_shape[axis]
        s = _expr.const(axis_size, dtype=ind_dtype)
    else:
        s = _op.take(_op.shape_of(data, dtype=ind_dtype), _op.const(axis, dtype="int64"))
    cond = fold_constant(indices < _op.const(0, ind_dtype))
    if isinstance(cond, _expr.Constant):
        val = cond.data.numpy()
        if val.size == 1:
            cond = val.item()
            if cond:
                indices = indices + s
            return fold_constant(indices)
    indices = _op.where(cond, indices + s, indices)
    return indices


class GatherOpt(Gather):
    """Operator converter for Gather with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get("axis", 0)
        cond, res = immediate_compute(np.take, inputs, **{"axis": axis})
        if cond:
            return res
        data = inputs[0]
        indices = inputs[1]
        indices = normalize_gather_indices(data, indices, axis, attr)
        return _op.take(data, indices, axis)


class GatherElementsOpt(GatherElements):
    """Operator converter for GatherElements with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        data = inputs[0]
        indices = inputs[1]
        axis = attr.get("axis", 0)
        indices = normalize_gather_indices(data, indices, axis, attr)
        return _op.gather(data, axis, indices)


class ElemwiseOpt(Elemwise):
    """A helper class for elemwise op converters."""

    name = ""
    func = ""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, f"Math op {cls.name} take 2 inputs, {len(inputs)} given"
        op_name = cls.name
        conv_ops = ["conv2d", "conv2d_transpose"]
        if attr.get("broadcast", 0) and any(x in str(inputs[0]) for x in conv_ops):
            # TODO(zhreshold): remove hard coded infershape
            axis = int(attr.get("axis", 0))
            inputs[1] = _op.expand_dims(inputs[1], axis=axis, num_newaxis=2)

        np_func = cls.func
        if (
            op_name == "divide"
            and isinstance(inputs[0], _expr.Constant)
            and isinstance(inputs[1], _expr.Constant)
        ):
            if "int" in inputs[0].data.dtype and "int" in inputs[1].data.dtype:
                # use np.divide is also ok! const->int
                np_func = np.floor_divide
        cond, res = immediate_compute(np_func, inputs, **{})
        if cond:
            return res
        return get_relay_op(op_name)(*inputs)


class AddOpt(ElemwiseOpt):
    """Operator converter for Add with optimization."""

    name = "add"
    func = np.add


class SubOpt(ElemwiseOpt):
    """Operator converter for Subtract with optimization."""

    name = "subtract"
    func = np.subtract


class MulOpt(ElemwiseOpt):
    """Operator converter for Multiply with optimization."""

    name = "multiply"
    func = np.multiply


class DivOpt(ElemwiseOpt):
    """Operator converter for Divide with optimization."""

    name = "divide"
    func = np.divide


class AndOpt(ElemwiseOpt):
    """Operator converter for And with optimization."""

    name = "logical_and"
    func = np.logical_and


class OrOpt(ElemwiseOpt):
    """Operator converter for Or with optimization."""

    name = "logical_or"
    func = np.logical_or


class NotOpt(Not):
    """Operator converter for Not with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        cond, res = immediate_compute(np.logical_not, [inputs[0]], **{})
        if cond:
            return res
        return _op.logical_not(inputs[0])


class EqualOpt(ElemwiseOpt):
    """Operator converter for Equal with optimization."""

    name = "equal"
    func = np.equal


class WhereOpt(Where):
    """Operator converter for Where with optimization."""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        cond, res = immediate_compute(np.where, inputs, is_where=True, **{})
        if cond:
            return res
        return _op.where(*inputs)


class TileOpt(ElemwiseOpt):
    """Operator converter for Tile with optimization."""

    name = "tile"
    func = np.tile


class RoundOpt(Round):
    """Operator converter for Round for simple.
    Note: ONNX/numpy: rounds to the nearest even value
    """

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        cond, res = immediate_compute(np.round, [inputs[0]], **{})
        if cond:
            return res
        return _op.round(inputs[0])


class Erf(OnnxOpConverter):
    """Operator converter for Erf with optimization"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        out = _op.erf(inputs[0])
        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)
        return out


class Identity(OnnxOpConverter):
    """Operator converter for Identity."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert isinstance(
            inputs[0], _expr.Constant
        ), "the type of data for Identity must be _expr.Constant!"
        return inputs[0]


class GlobalMaxPoolOpt(GlobalMaxPool):
    """Operator converter for GlobalMaxPool with optimization"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            data_shape = attr["inferred_shape"][0]
            rank = len(data_shape)
        else:
            rank = len(infer_shape(inputs[0]))
        if rank == 3:
            out = _op.nn.global_max_pool1d(inputs[0])
        elif rank == 4:
            out = _op.nn.global_max_pool2d(inputs[0])
        elif rank == 5:
            out = _op.nn.global_max_pool3d(inputs[0])
        else:
            raise NotImplementedError(
                "Global max pooling is only implemented for 1D, 2D, and 3D kernels, got %dD."
                % (rank - 2)
            )
        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)

        return out


class GlobalAveragePoolOpt(GlobalAveragePool):
    """Operator converter for GlobalAveragePool with optimization"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            data_shape = attr["inferred_shape"][0]
            rank = len(data_shape)
        else:
            rank = len(infer_shape(inputs[0]))
        if rank == 3:
            out = _op.nn.global_avg_pool1d(inputs[0])
        elif rank == 4:
            out = _op.nn.global_avg_pool2d(inputs[0])
        elif rank == 5:
            out = _op.nn.global_avg_pool3d(inputs[0])
        else:
            raise NotImplementedError(
                "Global average pooling is only implemented for 1D, 2D, and 3D kernels, got %dD."
                % (rank - 2)
            )
        if isinstance(inputs[0], _expr.Constant):
            out = fold_constant(out)

        return out


class PadOpt(Pad):
    """Operator converter for Pad with optimization."""

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        pads = inputs[1]
        if len(inputs) == 3 and inputs[2] is not None:
            if isinstance(inputs[2], _expr.Constant):
                value = np.take(inputs[2].data.numpy(), 0)
            else:
                value = fold_constant(_op.take(inputs[2], _op.const(0)))
        else:
            value = 0.0

        if isinstance(pads, _expr.Constant):
            pad_width_np = np.transpose(np.reshape(pads.data.numpy(), (2, -1)))
            pad_width = [list(i) for i in pad_width_np]
        else:
            pad_width = fold_constant(_op.transpose(_op.reshape(pads, (2, -1))))
        pad_mode = attr.get("mode", b"constant").decode("utf-8")
        if not pad_mode in ["constant", "edge", "reflect"]:
            raise tvm.error.OpAttributeInvalid(
                "Value " + pad_mode + ' in attribute "mode" is invalid for operator Pad.'
            )

        return _op.nn.pad(inputs[0], pad_width, value, pad_mode=pad_mode)


class UpsampleOpt(Upsample):
    """Operator converter for Upsample (nearest mode) with optimization."""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        scales = attr.get("scales")

        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            input_shape = attr["inferred_shape"][0]
        else:
            input_shape = infer_shape(inputs[0])
        dims = len(input_shape)

        if not scales:
            # Here we are going to higher OPSET version.
            assert len(inputs) == 2, f"Upsample op takes 2 inputs, {len(inputs)} given"

            if get_name(inputs[1]) in params:
                scales = params[inputs[1].name_hint].numpy()
            else:
                scales = inputs[1]
        if isinstance(scales, _expr.Constant):
            scales = list(scales.data.numpy())
        if not isinstance(scales, _expr.Expr):
            assert scales[0] == 1.0 and scales[1] == 1.0

        mode = attr.get("mode")
        if mode == b"nearest":
            method = "nearest_neighbor"
        elif mode == b"linear":
            method = "trilinear" if dims == 5 else "bilinear"
        else:
            raise tvm.error.OpAttributeInvalid(
                f'Value {mode} in attribute "mode" of operator Upsample is not valid.'
            )

        # in 3d case, we use the purely static op
        if dims == 5:
            if isinstance(scales, _expr.Expr):
                scale_h = _op.take(scales, _op.const(3))
                scale_w = _op.take(scales, _op.const(4))
                scale_d = _op.take(scales, _op.const(1))
            else:
                assert len(scales) == 5
                scale_h = scales[-2]
                scale_w = scales[-1]
                scale_d = scales[-3]

            layout = "NCDHW"
            out = _op.nn.upsampling3d(
                inputs[0],
                scale_d,
                scale_h,
                scale_w,
                layout=layout,
                method=method,
                coordinate_transformation_mode="asymmetric",
            )
        # in 2d case, use dynamic op
        else:
            if isinstance(scales, _expr.Expr):
                scale_h = _op.take(scales, _op.const(3))
                scale_w = _op.take(scales, _op.const(4))
            else:
                assert len(scales) == 4
                scale_h = scales[-2]
                scale_w = scales[-1]
            layout = "NCHW"

            out = _op.nn.upsampling(
                inputs[0], scale_h, scale_w, layout=layout, method=method, align_corners=False
            )
        return out


class ConstantOfShapeOpt(ConstantOfShape):
    """Operator converter for ConstantOfShape with optimization."""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        if "value" in attr:
            np_value = get_numpy(attr.pop("value"))[0]
            value = _expr.const(np_value)
            dtype = np_value.dtype.name
        else:
            value = _expr.const(0)
            dtype = "float32"
        if isinstance(inputs[0], _expr.Constant):
            shape = tuple(inputs[0].data.numpy())
            fill_value = value.data.numpy()
            return _expr.const(np.full(shape, fill_value=fill_value), dtype=dtype)
        output = _op.full(value, inputs[0], dtype=dtype)
        return output


class CastOpt(Cast):
    """Operator converter for Cast with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if isinstance(inputs[0], _expr.Constant):
            return _expr.const(inputs[0].data.numpy().astype(attr["to"]), dtype=attr["to"])
        return AttrCvt(op_name="cast", transforms={"to": "dtype"})(inputs, attr)

    @classmethod
    def _impl_v6(cls, inputs, attr, params):
        try:
            from onnx import TensorProto
        except ImportError as e:
            raise ImportError(f"Unable to import TensorProto from onnx {e}")

        # If onnx mapping is used, bfloat16 gets converted to float16
        # which is not the desired behavior
        if attr["to"] == int(TensorProto.BFLOAT16):
            attr["to"] = "bfloat16"
        else:
            try:
                from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

                attr["to"] = str(TENSOR_TYPE_TO_NP_TYPE[attr["to"]])
            except ImportError as e:
                raise ImportError(f"Unable to import onnx.mapping which is required {e}")
        if isinstance(inputs[0], _expr.Constant):
            return _expr.const(inputs[0].data.numpy().astype(attr["to"]), dtype=attr["to"])

        return AttrCvt(op_name="cast", transforms={"to": "dtype"})(inputs, attr)


class ReciprocalOpt(Reciprocal):
    """Operator converter for Reciprocal with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        cond, res = immediate_compute(np.reciprocal, inputs, **{})
        if cond:
            return res

        if "inferred_dtype" in attr and attr["inferred_dtype"][0]:
            dtype = attr["inferred_dtype"][0]
        else:
            dtype = infer_type(inputs[0]).checked_type.dtype

        return _expr.const(1.0, dtype=dtype) / inputs[0]


class RangeOpt(Range):
    """Operator converter for Range with optimization"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if len(inputs) != 3:
            raise ValueError("Expect 3 input only")

        if "inferred_dtype" in attr and attr["inferred_dtype"][0]:
            dtype = attr["inferred_dtype"][0]
        else:
            dtype = infer_type(inputs[0]).checked_type.dtype

        cond, res = immediate_compute(np.arange, inputs, **{"dtype": dtype})
        if cond:
            return res

        return _op.arange(inputs[0], inputs[1], inputs[2], dtype=dtype)


class SplitOpt(Split):
    """Operator converter for Split with optimization."""

    @classmethod
    def get_split_shape(cls, inputs, attr):
        """get split data shape"""
        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][0]
            and nonzero_in_it(attr["inferred_shape"][0])
        ):
            data_shape = attr["inferred_shape"][0]
        else:
            data_infer = infer_type(inputs[0])
            if not _ty.is_dynamic(data_infer.checked_type):
                data_shape = list(get_const_tuple(data_infer.checked_type.shape))
            else:
                data_shape = None
        return data_shape

    @classmethod
    def check_indices(cls, indices, split_dim):
        """check indices"""
        is_update = False
        reduce_split_num = 0
        if split_dim and isinstance(indices, list) and split_dim in indices:
            warnings.warn("there is 0 in split length!")
            is_update = True
            new_indices = [item for item in indices if item != split_dim]
            reduce_split_num = len(indices) - len(new_indices)
            if not new_indices:
                new_indices = 1
        else:
            new_indices = indices
        return new_indices, is_update, reduce_split_num

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        splits = attr.get("split", None)
        axis = attr.get("axis", 0)
        split_dim = None
        if splits is not None and len(splits) > 1:
            indices = []
            index = 0
            for i in splits[:-1]:
                index += i
                indices.append(index)
            data_shape = cls.get_split_shape(inputs, attr)
            split_dim = data_shape[axis] if data_shape else None
        # When splits isnt specified divide evenly over axis.
        else:
            indices = attr["tvm_custom"]["num_outputs"]
        cond, res = immediate_compute(
            np.split,
            [inputs[0]],
            is_split=True,
            **{"indices_or_sections": indices, "axis": axis},
        )
        if cond:
            output = _expr.TupleWrapper(_expr.Tuple(res), len(res))
        else:
            indices, is_update, reduce_split_num = cls.check_indices(indices, split_dim)
            output = _op.split(inputs[0], indices, axis)
            # if is_update:  # will raise other error!
            #     res = [output.astuple()]
            #     for i in range(reduce_split_num):
            #         res.append(_expr.const([]))
            #     output = _expr.TupleWrapper(_expr.Tuple(res), len(splits))
            attr["is_update"] = is_update
        # If the output of split is a single value, unpack if from the TupleWrapper
        if len(output) == 1:
            output = output[0]
        return output

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        splits = inputs[1]
        axis = attr.get("axis", 0)
        split_dim = None
        splits_rank = None
        if splits is not None:
            splits_rank = len(infer_shape(splits))
        if splits is not None and splits_rank > 0:
            if isinstance(splits, _expr.Constant):
                splits = splits.data.asnumpy()
                indices = []
                index = 0
                for i in splits[:-1]:
                    index += i
                    indices.append(index)
                data_shape = cls.get_split_shape(inputs, attr)
                split_dim = data_shape[axis] if data_shape else None
            else:
                raise ValueError("Dynamic Split not yet supported")
        # When splits isnt specified divide evenly over axis.
        else:
            indices = attr["tvm_custom"]["num_outputs"]
        cond, res = immediate_compute(
            np.split, [inputs[0]], **{"indices_or_sections": indices, "axis": axis}
        )
        if cond:
            output = TupleWrapper(res, len(res))
        else:
            indices, is_update, _ = cls.check_indices(indices, split_dim)
            output = _op.split(inputs[0], indices, axis)
            attr["is_update"] = is_update
        # If the output of split is a single value, unpack if from the TupleWrapper
        if len(output) == 1:
            output = output[0]
        return output


class ScatterElementsOpt(ScatterElements):
    """Operator converter for ScatterElements(fix bug)."""

    @classmethod
    def _args_check(cls, inputs, attr, red_valids=None):
        ret = []
        assert (
            len(inputs) == 3
        ), f"ScatterElements takes 3 inputs (data, indices, updates), {len(inputs)} given"
        if "inferred_dtype" in attr and attr["inferred_dtype"][1]:
            inp1_dtype = attr["inferred_dtype"][1]
        else:
            inp1_dtype = infer_type(inputs[1]).checked_type.dtype
        assert inp1_dtype in ["int32", "int64"]

        axis = attr.get("axis", 0)
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            rank = len(attr["inferred_shape"][0])
        else:
            rank = len(infer_shape(inputs[0]))
        assert rank > 0, "Data rank higher than 0 is expected"
        assert -rank <= axis < rank, "Axis is out of bounds"
        ret.append(axis)

        if red_valids:
            reduction = attr.get("reduction", None)
            if reduction is None:
                reduction = b"update"
            reduction = reduction.decode("utf-8")
            assert (
                reduction in red_valids
            ), f"Only {red_valids} modes are supported, but {reduction} is gotten"
            ret.append(reduction)

        return ret

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        ret = cls._args_check(inputs, attr)
        assert len(ret) == 1, f"There is a axis for ScatterElements! but {len(ret)}"
        axis = ret[0]
        return _op.scatter_elements(inputs[0], inputs[1], inputs[2], axis, "update")

    @classmethod
    def _impl_v16(cls, inputs, attr, params):
        axis, reduction = cls._args_check(inputs, attr, ["update", "add", "mul"])

        return _op.scatter_elements(inputs[0], inputs[1], inputs[2], axis, reduction)

    @classmethod
    def _impl_v18(cls, inputs, attr, params):
        axis, reduction = cls._args_check(inputs, attr, ["update", "add", "mul", "min", "max"])

        return _op.scatter_elements(inputs[0], inputs[1], inputs[2], axis, reduction)


class ScatterNDOpt(ScatterND):
    """Operator converter for ScatterND with optimization."""

    @classmethod
    def _inputs_check(cls, inputs, attr):
        if "inferred_dtype" in attr and attr["inferred_dtype"][1]:
            inp1_dtype = attr["inferred_dtype"][1]
        else:
            inp1_infer = infer_type(inputs[1])
            inp1_dtype = inp1_infer.checked_type.dtype
            inp1_shape = list(get_const_tuple(inp1_infer.checked_type.shape))
            if "inferred_shape" not in attr:
                attr["inferred_shape"] = [None, inp1_shape, None]
            else:
                attr["inferred_shape"][1] = inp1_shape

        assert (
            len(inputs) == 3
        ), f"ScatterND takes 3 inputs (data, indices, updates), {len(inputs)} given"
        assert inp1_dtype == "int64"

        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            data_rank = len(attr["inferred_shape"][0])
        else:
            data_rank = len(infer_shape(inputs[0]))
        assert data_rank > 0, "Data rank higher than 0 is expected"
        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][1]
            and nonzero_in_it(attr["inferred_shape"][1])
        ):
            inp1_shape = attr["inferred_shape"][1]
            indices_rank = len(inp1_shape)
        else:
            inp1_shape = infer_shape(inputs[1])
            indices_rank = len(inp1_shape)
            if "inferred_shape" not in attr:
                attr["inferred_shape"] = [None, inp1_shape, None]
            else:
                attr["inferred_shape"][1] = inp1_shape
        assert indices_rank > 0, "Indices rank higher than 0 is expected"
        if "inferred_shape" in attr and attr["inferred_shape"][2]:
            updates_rank = len(attr["inferred_shape"][2])
        else:
            updates_rank = len(infer_shape(inputs[2]))
        assert (
            updates_rank == data_rank + indices_rank - inp1_shape[-1] - 1
        ), "Updates rank should be equal to data_rank + indices_rank - indices_shape[-1] - 1"

    @classmethod
    def _reduction_check(cls, attr, red_valids=None):
        reduction = attr.get("reduction", None)
        if reduction is None:
            reduction = b"update"
        reduction = reduction.decode("utf-8")
        if red_valids is None:
            red_valids = ["update"]
        assert (
            reduction in red_valids
        ), f"Only {red_valids} reductions are supported, but {reduction} is gotten"

        return reduction

    @classmethod
    def _immediate_compute(cls, inputs, mod="update"):
        check_inputs = [isinstance(i, _expr.Constant) for i in inputs]
        cond = False
        res = None
        if all(check_inputs):
            cond = True
            const_inputs = [i.data.numpy() for i in inputs]
            data, indices, updates = const_inputs
            np_func_map = {"add": np.add, "mul": np.multiply, "min": np.minimum, "max": np.maximum}
            output = np.copy(data)
            update_indices = indices.shape[:-1]
            for idx in np.ndindex(update_indices):
                if mod == "update":
                    output[indices[idx]] = updates[idx]
                else:
                    output[indices[idx]] = np_func_map[mod](output[indices[idx]], updates[idx])
            out_dtype = inputs[0].data.dtype
            res = _expr.const(output, dtype=out_dtype)

        return cond, res

    @classmethod
    def _indices_adj(cls, inputs, attr, axes):
        """indices adjust"""
        if isinstance(inputs[1], _expr.Constant):
            indices_np = inputs[1].data.numpy()
            if np.any(indices_np < 0):
                if (
                    "inferred_shape" in attr
                    and attr["inferred_shape"][0]
                    and nonzero_in_it(attr["inferred_shape"][0])
                ):
                    data_shape = attr["inferred_shape"][0]
                else:
                    data_shape = infer_shape(inputs[0])
                for i in range(indices_np.shape[-1]):
                    indices_np[..., i] = np.where(
                        indices_np[..., i] < 0,
                        indices_np[..., i] + data_shape[i],
                        indices_np[..., i],
                    )
            indices = _expr.const(
                np.transpose(indices_np, tuple(axes[-1:] + axes[:-1])), inputs[1].data.dtype
            )
            indices_ = _expr.const(indices_np, inputs[1].data.dtype)
        else:
            # TODO: do for indices < 0(framework can work!): It is not efficient!
            # can be implemented by _op.slice + _op.where + _op.concatenate, then _op.transpose

            # if "inferred_shape" in attr and attr["inferred_shape"][0] and \
            #     nonzero_in_it(attr["inferred_shape"][0]):
            #     data_shape = attr["inferred_shape"][0]
            # else:
            #     data_shape = infer_shape(inputs[0])

            indices = _op.transpose(inputs[1], axes[-1:] + axes[:-1])
            indices_ = inputs[1]

        return indices, indices_

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        cls._inputs_check(inputs, attr)
        if "inferred_shape" in attr and attr["inferred_shape"][1]:
            indices_dim = len(attr["inferred_shape"][1])
        else:
            indices_dim = len(infer_shape(inputs[1]))
        axes = list(range(indices_dim))
        indices, indices_ = cls._indices_adj(inputs, attr, axes)
        # cond, res = cls._immediate_compute([inputs[0], indices_, inputs[2]])  # check
        # if cond:
        #     return res
        return _op.scatter_nd(inputs[0], indices, inputs[2])

    @classmethod
    def _impl_v16(cls, inputs, attr, params):
        cls._inputs_check(inputs, attr)
        reduction = cls._reduction_check(attr, ["update", "add", "mul"])

        if "inferred_shape" in attr and attr["inferred_shape"][1]:
            indices_dim = len(attr["inferred_shape"][1])
        else:
            indices_dim = len(infer_shape(inputs[1]))
        axes = list(range(indices_dim))
        indices, _ = cls._indices_adj(inputs, attr, axes)
        return _op.scatter_nd(inputs[0], indices, inputs[2], reduction)

    @classmethod
    def _impl_v18(cls, inputs, attr, params):
        cls._inputs_check(inputs, attr)
        reduction = cls._reduction_check(attr, ["update", "add", "mul", "min", "max"])

        if "inferred_shape" in attr and attr["inferred_shape"][1]:
            indices_dim = len(attr["inferred_shape"][1])
        else:
            indices_dim = len(infer_shape(inputs[1]))
        axes = list(range(indices_dim))
        indices, _ = cls._indices_adj(inputs, attr, axes)
        return _op.scatter_nd(inputs[0], indices, inputs[2], reduction)


class ResizeOpt(Resize):
    """Operator converter for Resize with optimization"""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        mode = attr.get("mode").decode("ascii")
        if mode == "nearest":
            method = "nearest_neighbor"
        elif mode == "linear":
            method = "linear"
        elif mode == "cubic":
            method = "cubic"
        else:
            raise tvm.error.OpAttributeInvalid(
                f'Value {mode} in attribute "mode" of operator Resize is not valid.'
            )

        scale = inputs[1]
        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][0]
            and nonzero_in_it(attr["inferred_shape"][0])
        ):
            data_shape = attr["inferred_shape"][0]
            ndims = len(data_shape)
            data_shape = _expr.const(data_shape, "int64")
        else:
            data_infer = infer_type(inputs[0])
            if not _ty.is_dynamic(data_infer.checked_type):
                data_shape = list(get_const_tuple(data_infer.checked_type.shape))
                ndims = len(data_shape)
                data_shape = _expr.const(data_shape, "int64")
            else:
                data_shape = shape_of(inputs[0])
                ndims = len(infer_shape(inputs[0]))

        scale_dtype = infer_type(scale).checked_type.dtype
        if isinstance(data_shape, _expr.Constant) and isinstance(scale, _expr.Constant):
            size = _expr.const(
                data_shape.data.numpy().astype(scale_dtype) * scale.data.numpy(), scale_dtype
            )
        else:
            size = _op.cast(data_shape, scale_dtype) * scale

        out = None
        if isinstance(size, _expr.Constant):
            out_size = _expr.const(size.data.numpy()[2:ndims], scale_dtype)
        else:
            out_size = fold_constant(_op.strided_slice(size, [2], [ndims]))

        if ndims == 3:
            out = _op.image.resize1d(inputs[0], out_size, None, "NCW", method, "asymmetric")
        elif ndims == 4:
            out = _op.image.resize2d(inputs[0], out_size, None, "NCHW", method, "asymmetric")
        elif ndims == 5:
            out = _op.image.resize3d(inputs[0], out_size, None, "NCDHW", method, "asymmetric")
        else:
            raise NotImplementedError("Resize only supports 3, 4, or 5 dims")

        if isinstance(inputs[0], _expr.Constant) and isinstance(out_size, _expr.Constant):
            return fold_constant(out)

        return out

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        if get_name(inputs[2]) in params:
            scale = _expr.const(params[inputs[2].name_hint])
        else:
            scale = inputs[2]
        scale_shape = infer_shape(scale)
        if len(inputs) == 4:
            assert (
                len(scale_shape) == 0 or scale_shape[0] == 0
            ), "One of scale or size should be passed, not both."
            size = inputs[3]
        else:
            if (
                "inferred_shape" in attr
                and attr["inferred_shape"][0]
                and nonzero_in_it(attr["inferred_shape"][0])
            ):
                data_shape = attr["inferred_shape"][0]
                data_shape = _expr.const(data_shape, "int64")
            else:
                data_infer = infer_type(inputs[0])
                if not _ty.is_dynamic(data_infer.checked_type):
                    data_shape = list(get_const_tuple(data_infer.checked_type.shape))
                    if "inferred_shape" not in attr:
                        attr["inferred_shape"] = [data_shape]
                    else:
                        attr["inferred_shape"][0] = data_shape
                    data_shape = _expr.const(data_shape, "int64")
                else:
                    data_shape = shape_of(inputs[0])

            assert len(scale_shape) != 0, "One of scale or size should be passed."
            scale_dtype = infer_type(scale).checked_type.dtype
            if isinstance(data_shape, _expr.Constant) and isinstance(scale, _expr.Constant):
                size = _expr.const(
                    data_shape.data.numpy().astype(scale_dtype) * scale.data.numpy(), scale_dtype
                )
            else:
                size = _op.cast(data_shape, scale_dtype) * scale

        return cls.v11_13_common(inputs, size, attr, params)

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        if get_name(inputs[2]) in params:
            scale = _expr.const(params[inputs[2].name_hint])
        else:
            scale = inputs[2]
        size = inputs[3]

        # Some versions of onnx exporters produce an opset 13 model with the opset 11
        # resize op, handle that edge case
        if scale is not None and size is not None:
            return cls._impl_v11(inputs, attr, params)

        if size is not None:
            assert scale is None, "One of scale or size should be passed, not both."
        else:
            scale_type = infer_type(scale)
            scale_shape = scale_type.checked_type.shape
            scale_dtype = scale_type.checked_type.dtype
            assert len(scale_shape) != 0, "One of scale or size should be passed."
            if (
                "inferred_shape" in attr
                and attr["inferred_shape"][0]
                and nonzero_in_it(attr["inferred_shape"][0])
            ):
                data_shape = attr["inferred_shape"][0]
                data_shape = _expr.const(data_shape, "int64")
            else:
                data_infer = infer_type(inputs[0])
                if not _ty.is_dynamic(data_infer.checked_type):
                    data_shape = list(get_const_tuple(data_infer.checked_type.shape))
                    if "inferred_shape" not in attr:
                        attr["inferred_shape"] = [data_shape]
                    else:
                        attr["inferred_shape"][0] = data_shape
                    data_shape = _expr.const(data_shape, "int64")
                else:
                    data_shape = shape_of(inputs[0])
            if isinstance(data_shape, _expr.Constant) and isinstance(scale, _expr.Constant):
                size = _expr.const(
                    data_shape.data.numpy().astype(scale_dtype) * scale.data.numpy(), scale_dtype
                )
            else:
                size = _op.cast(data_shape, scale_dtype) * scale

        return cls.v11_13_common(inputs, size, attr, params)

    @classmethod
    def v11_13_common(cls, inputs, size, attr, params):
        """
        Resize v11 and Resize v13 are identical except in how
        they handle the passing of scale and size. This utility
        provides the implementation for both
        """
        roi = inputs[1]
        if roi is not None and infer_shape(roi)[0] == 0:
            roi = None
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            ndims = len(attr["inferred_shape"][0])
        else:
            ndims = len(infer_shape(inputs[0]))
        mode = attr.get("mode").decode("ascii")
        if mode == "nearest":
            method = "nearest_neighbor"
        elif mode == "linear":
            method = "linear"
        elif mode == "cubic":
            method = "cubic"
        else:
            raise tvm.error.OpAttributeInvalid(
                f'Value {mode} in attribute "mode" of operator Resize is not valid.'
            )

        coord_trans = attr.get("coordinate_transformation_mode", b"half_pixel").decode("ascii")
        nearest_mode = attr.get("nearest_mode", b"round_prefer_floor").decode("ascii")
        alpha = attr.get("cubic_coeff_a", -0.75)
        exclude = attr.get("exclude_outside", 0)
        extrapolation_value = attr.get("extrapolation_value", 0.0)

        if roi is not None:
            if isinstance(roi, _expr.Constant):
                roi = _expr.const(
                    np.concatenate(
                        [
                            roi.data.numpy()[2:ndims],
                            roi.data.numpy()[(ndims + 2) : (2 * ndims)],
                        ],
                        axis=0,
                    ),
                    dtype=roi.data.dtype,
                )
            else:
                roi = fold_constant(
                    _op.concatenate(
                        [
                            _op.strided_slice(roi, [2], [ndims]),
                            _op.strided_slice(roi, [ndims + 2], [2 * ndims]),
                        ],
                        axis=0,
                    )
                )

        if isinstance(size, _expr.Constant):
            out_size = _expr.const(size.data.numpy()[2:ndims], size.data.dtype)
        else:
            out_size = fold_constant(_op.strided_slice(size, [2], [ndims]))

        out = None
        if ndims == 3:
            out = _op.image.resize1d(
                inputs[0],
                out_size,
                roi,
                "NCW",
                method,
                coord_trans,
                nearest_mode,
                alpha,
                exclude,
                extrapolation_value,
            )
        elif ndims == 4:
            out = _op.image.resize2d(
                inputs[0],
                out_size,
                roi,
                "NCHW",
                method,
                coord_trans,
                nearest_mode,
                alpha,
                exclude,
                extrapolation_value,
            )
        elif ndims == 5:
            out = _op.image.resize3d(
                inputs[0],
                out_size,
                roi,
                "NCDHW",
                method,
                coord_trans,
                nearest_mode,
                alpha,
                exclude,
                extrapolation_value,
            )
        else:
            raise NotImplementedError("Resize only supports 3, 4, or 5 dims")

        if (
            isinstance(inputs[0], _expr.Constant)
            and isinstance(out_size, _expr.Constant)
            and isinstance(roi, _expr.Constant)
        ):
            return fold_constant(out)

        return out


class LayerNormalizationOpt(LayerNormalization):
    """Operator converter for LayerNormalization from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v17(cls, inputs, attr, params):
        x = inputs[0]
        gamma = inputs[1]
        beta = inputs[2]
        axis = attr.get("axis", -1)
        eps = attr.get("epsilon", 1e-5)
        # according to the onnx doc, given the int axis (default -1)
        # to compute the mean and inv_stdev which are of dim [d[0], ..., d[axis-1], 1, ..., 1]
        # the actual computation is over (axis, ..., rank(x) - 1) axes
        # see https://github.com/onnx/onnx/blob/main/docs/Changelog.md#layernormalization-17
        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            rank = len(attr["inferred_shape"][0])
        else:
            checked_type = infer_type(x).checked_type
            rank = len(checked_type.shape)
            input_dtype = checked_type.dtype
            if "inferred_dtype" not in attr:
                attr["inferred_dtype"] = [input_dtype]
            else:
                attr["inferred_dtype"][0] = input_dtype
        if "inferred_dtype" in attr and attr["inferred_dtype"][0]:
            dtype = attr["inferred_dtype"][0]
        else:
            dtype = infer_type(x).checked_type.dtype
        axis = tuple(range(axis, rank)) if axis >= 0 else tuple(range(rank + axis, rank))
        mean = _op.mean(x, axis, keepdims=True)
        var = _op.variance(x, axis, keepdims=True, with_mean=mean)
        inv_stdev = _op.divide(
            _op.const(1, dtype=dtype), _op.sqrt(_op.add(var, _op.const(eps, dtype=dtype)))
        )
        x_norm = _op.multiply(_op.subtract(x, mean), inv_stdev)
        ln = _op.multiply(x_norm, gamma)
        if beta is not None:
            ln = _op.add(ln, beta)

        if isinstance(x, _expr.Constant):
            ln = fold_constant(ln)
            mean = fold_constant(mean)
            inv_stdev = fold_constant(inv_stdev)

        return _expr.TupleWrapper(_expr.Tuple([ln, mean, inv_stdev]), 3)


class GroupNormalization(OnnxOpConverter):
    """Operator converter for LayerNormalization"""

    @classmethod
    def _impl_v18(cls, inputs, attr, params):
        # When the number of groups is the same as the number of channels, this operator is
        # equivalent to InstanceNormalization. GroupNorm is exported using a subgraph of ops
        # that is in ONNX. The subgraph is Reshape + InstanceNorm + Reshape + Mul + Add.
        # So we can not see GroupNorm in onnx model, however, the subgraph is able to fuse to
        # a GroupNorm for efficient processing.
        x = inputs[0]
        gamma = inputs[1]
        beta = inputs[2]
        eps = attr.get("epsilon", 1e-5)
        num_groups = attr["num_groups"]

        if "inferred_shape" in attr and attr["inferred_shape"][0]:
            data_shape = attr["inferred_shape"][0]
            rank = len(data_shape)
        else:
            checked_type = infer_type(x).checked_type
            data_shape = get_const_tuple(checked_type.shape)
            rank = len(data_shape)
            input_dtype = checked_type.dtype
            if "inferred_dtype" not in attr:
                attr["inferred_dtype"] = [input_dtype]
            else:
                attr["inferred_dtype"][0] = input_dtype
        if "inferred_dtype" in attr and attr["inferred_dtype"][0]:
            dtype = attr["inferred_dtype"][0]
            del attr["inferred_dtype"]
        else:
            dtype = infer_type(x).checked_type.dtype

        assert (
            data_shape[1] % num_groups == 0
        ), "channel number should be divide exactly by num_groups!"
        if "stash_type" in attr and attr["stash_type"] == 1 and dtype != "float32":
            warnings.warn(
                "The floating-point precision used in stage one of the computation when opset >= 21"
            )

        out = _op.nn.group_norm(
            x,
            gamma=gamma,
            beta=beta,
            num_groups=num_groups,
            axis=1,
            epsilon=eps,
            center=True,
            scale=True,
        )

        if isinstance(x, _expr.Constant):
            out = fold_constant(out)

        return out


class FlattenOpt(Flatten):
    """Operator converter for Flatten with optimization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if (
            "inferred_shape" in attr
            and attr["inferred_shape"][0]
            and nonzero_in_it(attr["inferred_shape"][0])
        ):
            data_shape = attr["inferred_shape"][0]
            ndim = len(data_shape)
            ishape = _expr.const(data_shape, "int64")
        else:
            data_infer = infer_type(inputs[0])
            if not _ty.is_dynamic(data_infer.checked_type):
                data_shape = list(get_const_tuple(data_infer.checked_type.shape))
                ndim = len(data_shape)
                ishape = _expr.const(data_shape, "int64")
            else:
                ishape = shape_of(inputs[0])
                ndim = infer_shape(ishape)[0]

        axis = attr.get("axis", 1)
        if axis < 0:
            axis = axis + ndim
        if isinstance(inputs[0], _expr.Constant):
            new_shape = (1, -1) if axis == 0 else (np.prod(ishape.data.numpy()[0:axis]), -1)
            cond, res = immediate_compute(np.reshape, inputs, **{"newshape": new_shape})
            if cond:
                return res

        if axis == 1:
            out = _op.nn.batch_flatten(inputs[0])
        else:
            if isinstance(ishape, _expr.Constant):
                newshape = [1, -1] if axis == 0 else [np.prod(ishape.data.numpy()[0:axis]), -1]
            else:
                pre_shape = _op.prod(_op.strided_slice(ishape, [0], [axis], [1]), keepdims=True)
                post_shape = _op.prod(_op.strided_slice(ishape, [axis], [ndim], [1]), keepdims=True)
                newshape = fold_constant(_op.concatenate([pre_shape, post_shape], axis=0))
            out = _op.reshape(inputs[0], newshape)
        return out


def get_scalar(x, params, dtype="float32"):
    """Helper to get a scalar value for Quantized operators."""
    if isinstance(x, _expr.Var) and x.name_hint in params:
        return _op.const(params[x.name_hint].numpy(), dtype)
    rank = len(infer_shape(x))
    assert rank <= 1, "scale and zero_point input must be scalars"
    if isinstance(x, _expr.Constant):
        if rank == 1:
            out = np.squeeze(x.data.numpy(), [0]).astype(dtype)
        else:
            out = x.data.numpy().astype(dtype)
        return _expr.const(out, dtype)
    if rank == 1:
        x = _op.squeeze(x, [0])
    return _op.cast(x, dtype)


def get_scalar_or_1d_tensor(x, params, dtype="float32"):
    """Helper to get a scalar value or 1D tensor for Quantized operators."""
    if isinstance(x, _expr.Var) and x.name_hint in params:
        return _op.const(params[x.name_hint].numpy(), dtype)
    rank = len(infer_shape(x))
    assert rank <= 1, "scale and zero_point input must be scalars or 1D tensors"
    if isinstance(x, _expr.Constant):
        return _expr.const(x.data.numpy().astype(dtype), dtype=dtype)
    return _op.cast(x, dtype)


def check_quantize_constraint(data=None, zp=None, id_name=""):
    """check input/zp dtype and value"""
    if data and isinstance(data, _expr.Constant):
        assert data.data.dtype == "int8", f"the dtype of {id_name} must be 'int8'!"

    if zp and isinstance(zp, _expr.Constant):
        assert zp.data.dtype == "int8", f"the zp type of {id_name} must be 'int8'!"
        rank = len(zp.data.shape)
        if rank == 0:
            value_check = zp.data.numpy() == 0
        else:
            value_check = (zp.data.numpy() == 0).all()
        assert (
            value_check
        ), f"Only support Symmetric quantization, and the zp value of {id_name} is not zero!"


class QuantizeLinearOpt(QuantizeLinear):
    """Operator converter for QuantizeLinear."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        data, scale, zp = inputs
        check_quantize_constraint(zp=zp, id_name="QuantizeLinear")

        out_dtype = infer_type(zp).checked_type.dtype
        if isinstance(zp, _expr.Constant):
            zp = _expr.const(zp.data.numpy().astype("int32"), "int32")
        else:
            zp = _op.cast(zp, "int32")
        axis = 0 if len(infer_shape(data)) <= 1 else 1
        return _qnn.op.quantize(data, scale, zp, axis, out_dtype)

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        data, scale, zp = inputs
        check_quantize_constraint(zp=zp, id_name="QuantizeLinear")

        out_dtype = infer_type(zp).checked_type.dtype
        axis = attr.get("axis", 1)
        if len(infer_shape(data)) <= 1:
            axis = 0
        if isinstance(zp, _expr.Constant):
            zp = _expr.const(zp.data.numpy().astype("int32"), "int32")
        else:
            zp = _op.cast(zp, "int32")
        return _qnn.op.quantize(data, scale, zp, axis, out_dtype)


class DequantizeLinearOpt(DequantizeLinear):
    """Operator converter for QuantizeLinear."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        data, scale, zp = inputs
        check_quantize_constraint(zp=zp, id_name="DequantizeLinear")

        if isinstance(zp, _expr.Constant):
            zp = _expr.const(zp.data.numpy().astype("int32"), "int32")
        else:
            zp = _op.cast(zp, "int32")
        axis = 0 if len(infer_shape(data)) <= 1 else 1
        return _qnn.op.dequantize(data, scale, zp, axis)

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        data, scale, zp = inputs
        check_quantize_constraint(zp=zp, id_name="DequantizeLinear")

        axis = attr.get("axis", 1)
        if len(infer_shape(data)) <= 1:
            axis = 0
        if isinstance(zp, _expr.Constant):
            zp = _expr.const(zp.data.numpy().astype("int32"), "int32")
        else:
            zp = _op.cast(zp, "int32")
        return _qnn.op.dequantize(data, scale, zp, axis)


class QLinearAddOpt(QLinearAdd):
    """Operator converter for QLinearAdd from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        a = inputs[0]
        a_scale = get_scalar(inputs[1], params)
        check_quantize_constraint(zp=inputs[2], id_name="QLinearAdd_a")
        a_zero_point = get_scalar(inputs[2], params, "int32")
        b = inputs[3]
        b_scale = get_scalar(inputs[4], params)
        check_quantize_constraint(zp=inputs[5], id_name="QLinearAdd_b")
        b_zero_point = get_scalar(inputs[5], params, "int32")
        c_scale = get_scalar(inputs[6], params)
        check_quantize_constraint(zp=inputs[7], id_name="QLinearAdd_c")
        c_zero_point = get_scalar(inputs[7], params, "int32")

        a_infer = infer_type(a).checked_type
        dtype = a_infer.dtype
        a_rank = len(a_infer.shape)
        b_rank = len(infer_type(b).checked_type.shape)
        a_axis = 0 if a_rank <= 1 else 1
        b_axis = 0 if b_rank <= 1 else 1

        if attr.get("use_float", None):
            a = _qnn.op.dequantize(a, a_scale, a_zero_point, a_axis)
            b = _qnn.op.dequantize(b, b_scale, b_zero_point, b_axis)
            out = _op.add(a, b)
            axis = a_axis if a_rank > b_rank else b_axis  # may broadcast
            return _qnn.op.quantize(out, c_scale, c_zero_point, axis, out_dtype=dtype)

        return edgex_add(
            a,
            b,
            a_scale,
            a_zero_point,
            b_scale,
            b_zero_point,
            c_scale,
            c_zero_point,
            rhs_axis=a_axis,
            lhs_axis=b_axis,
        )


class QLinearMulOpt(QLinearMul):
    """Operator converter for QLinearMul from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        a = inputs[0]
        a_scale = get_scalar(inputs[1], params)
        check_quantize_constraint(zp=inputs[2], id_name="QLinearMul_a")
        a_zero_point = get_scalar(inputs[2], params, "int32")
        b = inputs[3]
        b_scale = get_scalar(inputs[4], params)
        check_quantize_constraint(zp=inputs[5], id_name="QLinearMul_b")
        b_zero_point = get_scalar(inputs[5], params, "int32")
        y_scale = fold_constant(get_scalar(inputs[6], params))
        check_quantize_constraint(zp=inputs[7], id_name="QLinearMul_c")
        y_zero_point = get_scalar(inputs[7], params, "int32")

        a_rank = len(infer_type(a).checked_type.shape)
        b_rank = len(infer_type(b).checked_type.shape)
        a_axis = 0 if a_rank <= 1 else 1
        b_axis = 0 if b_rank <= 1 else 1

        return edgex_mul(
            a,
            b,
            a_scale,
            a_zero_point,
            b_scale,
            b_zero_point,
            y_scale,
            y_zero_point,
            lhs_axis=a_axis,
            rhs_axis=b_axis,
        )


class QLinearConvOpt(QLinearConv):
    """Operator converter for QLinearConv."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        data = inputs[0]
        x_scale = get_scalar(inputs[1], params)
        check_quantize_constraint(zp=inputs[2], id_name="QLinearConv_x")
        x_zero_point = get_scalar(inputs[2], params, "int32")
        check_quantize_constraint(data=inputs[3], zp=inputs[5], id_name="QLinearConv_w")
        weight = inputs[3]
        w_scale = get_scalar_or_1d_tensor(inputs[4], params)
        w_zero_point = get_scalar_or_1d_tensor(inputs[5], params, "int32")
        y_scale = fold_constant(get_scalar(inputs[6], params))
        check_quantize_constraint(zp=inputs[7], id_name="QLinearConv_y")
        y_zero_point = get_scalar(inputs[7], params, "int32")

        # Check shapes for per channel quantization
        w_scale_shape = infer_shape(w_scale)
        w_zero_point_shape = infer_shape(w_zero_point)
        if len(w_scale_shape) == 1 or len(w_zero_point_shape) == 1:
            m = infer_shape(weight)[0]
            if m != w_scale_shape[0] or m != w_zero_point_shape[0]:
                raise tvm.error.OpAttributeInvalid(
                    "The number of elements should be equal to the number of output channels"
                )

        input_shape = infer_shape(data)

        ndim = len(input_shape)
        kernel_type = infer_type(weight)
        kernel_shapes = [get_const_tuple(kernel_type.checked_type.shape)]
        if "kernel_shape" not in attr:
            attr["kernel_shape"] = kernel_shapes[0][2:]

        if "auto_pad" in attr:
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                # Warning: Convolution does not yet support dynamic shapes,
                # one will need to run dynamic_to_static on this model after import
                zp = fold_constant(x_zero_point)
                assert isinstance(zp, relay.Constant), "Zero point expected to be a constant"
                data = autopad(
                    data,
                    attr.get("strides", [1] * (ndim - 2)),
                    attr["kernel_shape"],
                    attr.get("dilations", [1] * (ndim - 2)),
                    pad_value=zp.data,
                    mode=attr["auto_pad"],
                )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = tuple([0 for i in range(ndim - 2)])
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = (
                    f'Value {attr["auto_pad"]} in attribute "auto_pad" of operator Conv '
                    f"is invalid."
                )
                raise tvm.error.OpAttributeInvalid(msg)
            attr.pop("auto_pad")

        out_channels = kernel_shapes[0][0]
        dilation = attr.get("dilations", [1] * (ndim - 2))
        strides = attr.get("strides", [1] * (ndim - 2))
        padding = attr["pads"] if "pads" in attr else 0
        groups = attr["group"] if "group" in attr else 1

        if ndim != 4 and ndim != 5:
            raise tvm.error.OpAttributeInvalid(
                "Only 2D or 3D kernels are supported for operator QLinearConv."
            )

        if ndim == 4:
            out = _qnn.op.conv2d(
                data,
                weight,
                x_zero_point,
                w_zero_point,
                x_scale,
                w_scale,
                kernel_size=attr["kernel_shape"],
                channels=out_channels,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        elif ndim == 5:
            out = edgex_conv3d(
                data,
                weight,
                x_zero_point,
                w_zero_point,
                x_scale,
                w_scale,
                kernel_size=attr["kernel_shape"],
                channels=out_channels,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        use_bias = len(inputs) == 9 and inputs[8]
        if use_bias:
            out = _op.nn.bias_add(out, inputs[8])

        out_dtype = infer_type(inputs[7]).checked_type.dtype
        if isinstance(x_scale, _expr.Constant) and isinstance(w_scale, _expr.Constant):
            requantize_scale = _expr.const(
                np.multiply(x_scale.data.numpy(), w_scale.data.numpy()), x_scale.data.dtype
            )
        else:
            requantize_scale = _op.multiply(x_scale, w_scale)

        # requantize requires y_scale to be constant,
        # if y_scale is not constant, doing dequantize -> quantize
        if isinstance(y_scale, _expr.Constant):
            out = _qnn.op.requantize(
                out,
                requantize_scale,
                _op.const(0, dtype="int32"),
                y_scale,
                y_zero_point,
                out_dtype=out_dtype,
                axis=1,
            )
        else:
            out = _qnn.op.dequantize(out, requantize_scale, _op.const(0, dtype="int32"), axis=1)
            out = _qnn.op.quantize(out, y_scale, y_zero_point, axis=1, out_dtype=out_dtype)
        return out


class QGemmOpt(QGemm):
    """Operator converter for QGemm."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QGemm

        a = inputs[0]
        a_scale = get_scalar(inputs[1], params)
        check_quantize_constraint(zp=inputs[2], id_name="QGemm_a")
        a_zp = get_scalar(inputs[2], params, "int32")

        b = inputs[3]
        # must be a scalar or 1D tensor which means a per-tensor or per-column quantization
        # If 1-D tensor, number of elements should be equal to columns elements of input B
        b_scale = get_scalar_or_1d_tensor(inputs[4], params)
        check_quantize_constraint(zp=inputs[5], id_name="QGemm_b")
        b_zp = get_scalar_or_1d_tensor(inputs[5], params, "int32")

        # note that if optional and not provided then value will be None.
        C = inputs[6]
        # must be null or a scalar or 1D tensor of size 1
        y_scale = inputs[7]
        # must be null or a scalar or 1D tensor of size 1
        check_quantize_constraint(zp=inputs[8], id_name="QGemm_y")
        y_zp = get_scalar(inputs[8], params, "int32")

        assert len(infer_shape(a)) == 2
        b_shape = infer_shape(b)
        assert len(b_shape) == 2
        # zero point and scale of input b should have same shape size
        b_scale_shape = infer_shape(b_scale)
        b_zp_shape = infer_shape(b_zp)
        # b_zp: default value
        if len(b_scale_shape) == 1 and len(b_zp_shape) == 0 and b_zp.data.numpy() == 0:
            b_zp = _expr.const(np.broadcast_to(0, b_scale_shape), dtype="int32")
            b_zp_shape = b_scale_shape
        assert b_scale_shape == b_zp_shape

        alpha = float(attr.get("alpha", 1.0))
        transA = int(attr.get("transA", 0))
        transB = int(attr.get("transB", 0))

        if len(b_scale_shape) == 1:
            n = b_shape[1] if not transB else b_shape[0]
            if n != b_scale_shape[0]:
                raise tvm.error.OpAttributeInvalid(
                    f"The number of elements should be equal to the number of output channels, "
                    f"but {b_scale_shape[0]} vs {n}"
                )
        # get number of channels
        channels = infer_channels(b, not transB)
        a_dtype = infer_type(a).checked_type.dtype

        if transA:
            if isinstance(a, _expr.Constant):
                a = _expr.const(np.transpose(a.data.numpy(), axes=(1, 0)), a.data.dtype)
            else:
                a = _op.transpose(a, axes=(1, 0))
        if not transB:
            if isinstance(b, _expr.Constant):
                b = _expr.const(np.transpose(b.data.numpy(), axes=(1, 0)), b.data.dtype)
            else:
                b = _op.transpose(b, axes=(1, 0))

        result = _qnn.op.dense(a, b, a_zp, b_zp, a_scale, b_scale, channels)

        if C:
            result = _op.add(result, C)

        if isinstance(a_scale, _expr.Constant) and isinstance(b_scale, _expr.Constant):
            requantize_scale = _expr.const(
                np.multiply(a_scale.data.numpy(), b_scale.data.numpy()), a_scale.data.dtype
            )
        else:
            requantize_scale = _op.multiply(a_scale, b_scale)
        if alpha != 1.0:
            if isinstance(requantize_scale, _expr.Constant):
                requantize_scale = _expr.const(
                    np.multiply(requantize_scale.data.numpy(), alpha), requantize_scale.data.dtype
                )
            else:
                requantize_scale *= _expr.const(alpha, dtype="float32")
        requantize_zp = _op.const(0, dtype="int32")

        if y_scale:
            # requantize requires y_scale to be constant,
            # if y_scale is not constant, doing dequantize -> quantize
            if isinstance(y_scale, _expr.Constant):
                y = _qnn.op.requantize(
                    result,
                    requantize_scale,
                    requantize_zp,
                    y_scale,
                    y_zp,
                    axis=-1,
                    rounding="TONEAREST",
                    out_dtype=a_dtype,
                )
            else:
                result_deq = _qnn.op.dequantize(result, requantize_scale, requantize_zp, axis=0)

                y = _qnn.op.quantize(result_deq, y_scale, y_zp, axis=0, out_dtype=a_dtype)
        else:
            y = _op.multiply(_op.cast(result, "float32"), requantize_scale)

        return y


class QLinearLeakyReluOpt(QLinearLeakyRelu):
    """Operator converter for QLinearLeakyRelu from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        a_scale = get_scalar(inputs[1], params)
        check_quantize_constraint(zp=inputs[2], id_name="QLinearLeakyRelu_x")
        a_zero_point = get_scalar(inputs[2], params, "int32")
        y_scale = fold_constant(get_scalar(inputs[3], params))
        check_quantize_constraint(zp=inputs[4], id_name="QLinearLeakyRelu_y")
        y_zero_point = get_scalar(inputs[4], params, "int32")
        alpha = float(attr.get("alpha", 0.01))  # the same as float leaky_relu

        input_infer = infer_type(inputs[0]).checked_type
        dtype = input_infer.dtype
        rank = len(input_infer.shape)
        axis = 0 if rank <= 1 else 1

        # Onnxruntime doesn't actually do this op in integer, they dequantize to fp32
        # and then requantize afer (according to documentation below)
        # https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.QLinearLeakyRelu
        if attr.get("use_float", None):
            a = _qnn.op.dequantize(inputs[0], a_scale, a_zero_point, axis)
            out = _op.nn.leaky_relu(a, alpha)
            return _qnn.op.quantize(out, y_scale, y_zero_point, axis, out_dtype=dtype)
        return edgex_leaky_relu(
            inputs[0], a_scale, a_zero_point, y_scale, y_zero_point, alpha, dtype, axis
        )


class QLinearSigmoidOpt(QLinearSigmoid):
    """Operator converter for QLinearSigmoid from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        x = inputs[0]
        x_scale = get_scalar(inputs[1], params)
        x_zero_point = get_scalar(inputs[2], params, "int32")
        y_scale = fold_constant(get_scalar(inputs[3], params))
        y_zero_point = get_scalar(inputs[4], params, "int32")

        x_infer = infer_type(x).checked_type
        dtype = x_infer.dtype
        rank = len(x_infer.shape)
        axis = 0 if rank <= 1 else 1

        ## Apparently, onnxruntime doesn't do this op in integer, they dequantize to fp32
        ## and then requantize after:
        ## https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/
        ## providers/dml/DmlExecutionProvider/src/GraphTransformer.cpp#L245
        x = _qnn.op.dequantize(x, x_scale, x_zero_point, axis)
        out = _op.sigmoid(x)
        return _qnn.op.quantize(out, y_scale, y_zero_point, axis, out_dtype=dtype)


class QLinearSoftmaxOpt(QLinearSoftmax):
    """Operator converter for QLinearSoftmax from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr["axis"]

        x = inputs[0]
        x_scale = get_scalar(inputs[1], params)
        x_zero_point = get_scalar(inputs[2], params, "int32")
        y_scale = fold_constant(get_scalar(inputs[3], params))
        y_zero_point = get_scalar(inputs[4], params, "int32")

        x_infer = infer_type(x).checked_type
        dtype = x_infer.dtype
        rank = len(x_infer.shape)
        q_axis = 0 if rank <= 1 else 1

        x = _qnn.op.dequantize(x, x_scale, x_zero_point, q_axis)
        out = _op.nn.softmax(x, axis)
        return _qnn.op.quantize(out, y_scale, y_zero_point, q_axis, out_dtype=dtype)


class QLinearConcatOpt(QLinearConcat):
    """Operator converter for QLinearConcat from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # which axis to concat on
        axis = attr["axis"]

        y_scale = fold_constant(get_scalar(inputs[0], params))
        check_quantize_constraint(zp=inputs[1], id_name="QLinearConcat_y")
        y_zero_point = get_scalar(inputs[1], params, "int32")

        # input tensors, scales, zero_points
        assert (
            len(inputs) % 3 == 2
        ), "Additional input count must be a multiple of 3 -- tensor/scale/zero_point tuples"
        tensors = []
        scales = []
        zero_points = []
        for i in range(2, len(inputs), 3):
            tensors.append(inputs[i])
            scales.append(get_scalar(inputs[i + 1], params))
            zero_points.append(get_scalar(inputs[i + 2], params, "int32"))

        return _qnn.op.concatenate(tensors, scales, zero_points, y_scale, y_zero_point, axis)


def qmatmul(
    a,
    b,
    a_zp_scalar,
    b_zp_scalar,
    a_scale_scalar,
    b_scale_scalar,
    transform_num_hidden_units,
    matmul_result_dtype,
    a_shape,
    a_rank,
    b_shape,
    b_rank,
    b_type,
):
    """
    Helper function to handle QLinearMatMul
    It is very close to 'matmul_out_dtype' but separated due to
    differences in signatures of dense, matmul, batch_matmul of nn and qnn.
    They requre scaling and zero point arguments
    """
    # a_shape = shape_of(a)
    # a_rank = infer_shape(a_shape)[0]
    # b_shape = shape_of(b)
    # b_rank = infer_shape(b_shape)[0]
    if a_rank > 2 or b_rank > 2:
        # Determine the output batch dimension.
        new_a_shape = a_shape
        new_b_shape = b_shape
        if a_rank > b_rank:
            rank_diff = a_rank - b_rank
            if isinstance(b_shape, _expr.Constant):
                new_b_shape = _expr.const(
                    np.concatenate(
                        [
                            [1] * rank_diff,
                            b_shape.data.numpy(),
                        ],
                        axis=0,
                    ),
                    dtype="int64",
                )
            else:
                new_b_shape = _op.concatenate(
                    [  # #infer_type(b_shape).checked_type.dtype -> dtype
                        _expr.const([1] * rank_diff, dtype="int64"),
                        b_shape,
                    ],
                    0,
                )
        elif a_rank < b_rank:
            rank_diff = b_rank - a_rank
            if isinstance(a_shape, _expr.Constant):
                new_a_shape = _expr.const(
                    np.concatenate(
                        [
                            [1] * rank_diff,
                            a_shape.data.numpy(),
                        ],
                        axis=0,
                    ),
                    dtype="int64",
                )
            else:
                new_a_shape = _op.concatenate(
                    [  # infer_type(a_shape).checked_type.dtype -> dtype
                        _expr.const([1] * rank_diff, dtype="int64"),
                        a_shape,
                    ],
                    0,
                )
        else:
            pass

        if isinstance(new_a_shape, _expr.Constant) and isinstance(new_b_shape, _expr.Constant):
            batch_end = max(a_rank, b_rank) - 2
            out_batch = _expr.const(
                np.concatenate(
                    [
                        np.maximum(
                            new_b_shape.data.numpy()[0:batch_end],
                            new_a_shape.data.numpy()[0:batch_end],
                        ),
                    ],
                    axis=0,
                ),
                dtype="int64",
            )
        else:
            # out_batch = _op.concatenate(
            #     [
            #         _op.maximum(
            #             _op.strided_slice(new_b_shape, [i], [i + 1]),
            #             _op.strided_slice(new_a_shape, [i], [i + 1]),
            #         )
            #         for i in range(max(a_rank, b_rank) - 2)
            #     ],
            #     0,
            # )
            batch_end = max(a_rank, b_rank) - 2
            out_batch = _op.concatenate(
                [
                    _op.maximum(
                        _op.strided_slice(new_b_shape, [0], [batch_end]),
                        _op.strided_slice(new_a_shape, [0], [batch_end]),
                    )
                ],
                0,
            )

        # Convert to dense if the second matrix is 2d and non-dynamic
        if b_rank == 2 and not _ty.is_dynamic(b_type):
            a = flatten_to_nd(a, a_shape, 2)
            if isinstance(b, _expr.Constant):
                b = _expr.const(np.transpose(b.data.numpy()), dtype=b.data.dtype)
            else:
                b = _op.transpose(b)
            output = _qnn.op.dense(
                a,
                b,
                a_zp_scalar,
                b_zp_scalar,
                a_scale_scalar,
                b_scale_scalar,
                transform_num_hidden_units,
                matmul_result_dtype,
            )
        else:
            # broadcast a and b
            if isinstance(out_batch, _expr.Constant) and isinstance(a_shape, _expr.Constant):
                a_broadcasted_shape = _expr.const(
                    np.concatenate(
                        [out_batch.data.numpy(), a_shape.data.numpy()[a_rank - 2 : a_rank]], axis=0
                    ),
                    dtype="int64",
                )
            else:
                a_broadcasted_shape = fold_constant(
                    _op.concatenate(
                        [
                            out_batch,
                            fold_constant(_op.strided_slice(a_shape, [a_rank - 2], [a_rank])),
                        ],
                        0,
                    )
                )
            if isinstance(out_batch, _expr.Constant) and isinstance(b_shape, _expr.Constant):
                b_broadcasted_shape = _expr.const(
                    np.concatenate(
                        [out_batch.data.numpy(), b_shape.data.numpy()[b_rank - 2 : b_rank]], axis=0
                    ),
                    dtype="int64",
                )
            else:
                b_broadcasted_shape = fold_constant(
                    _op.concatenate(
                        [
                            out_batch,
                            fold_constant(_op.strided_slice(b_shape, [b_rank - 2], [b_rank])),
                        ],
                        0,
                    )
                )

            if not tvm.ir.structural_equal(a_shape, a_broadcasted_shape):
                if isinstance(a, _expr.Constant) and isinstance(
                    a_broadcasted_shape, _expr.Constant
                ):
                    a = _expr.const(
                        np.broadcast_to(a.data.numpy(), tuple(a_broadcasted_shape.data.numpy())),
                        dtype=a.data.dtype,
                    )
                else:
                    a = _op.transform.broadcast_to(a, a_broadcasted_shape)
            if not tvm.ir.structural_equal(b_shape, b_broadcasted_shape):
                if isinstance(b, _expr.Constant) and isinstance(
                    b_broadcasted_shape, _expr.Constant
                ):
                    b = _expr.const(
                        np.broadcast_to(b.data.numpy(), tuple(b_broadcasted_shape.data.numpy())),
                        dtype=b.data.dtype,
                    )
                else:
                    b = _op.transform.broadcast_to(b, b_broadcasted_shape)

            # Convert a and b into 3 dimensional tensors.
            # a = flatten_to_nd(a, shape_of(a), 3)
            # b = flatten_to_nd(b, shape_of(b), 3)
            a = flatten_to_nd(a, a_broadcasted_shape, 3)
            b = flatten_to_nd(b, b_broadcasted_shape, 3)

            # Transpose matrix dimensions of b.
            if isinstance(b, _expr.Constant):
                bt = _expr.const(np.transpose(b.data.numpy(), (0, 2, 1)), dtype=b.data.dtype)
            else:
                bt = _op.transpose(b, [0, 2, 1])
            # Perform a NT batch matmul.
            output = _qnn.op.batch_matmul(
                a, bt, a_zp_scalar, b_zp_scalar, a_scale_scalar, b_scale_scalar, matmul_result_dtype
            )
        # Reshape output to original dimensions.
        if (
            isinstance(out_batch, _expr.Constant)
            and isinstance(a_shape, _expr.Constant)
            and isinstance(b_shape, _expr.Constant)
        ):
            final_shape = _expr.const(
                np.concatenate(
                    [
                        out_batch.data.numpy(),
                        a_shape.data.numpy()[a_rank - 2 : a_rank - 1],
                        b_shape.data.numpy()[b_rank - 1 : b_rank],
                    ],
                    axis=0,
                ),
                dtype="int64",
            )
        else:
            final_shape = _op.concatenate(
                [
                    out_batch,
                    fold_constant(_op.strided_slice(a_shape, [a_rank - 2], [a_rank - 1])),
                    fold_constant(_op.strided_slice(b_shape, [b_rank - 1], [b_rank])),
                ],
                0,
            )
            final_shape = fold_constant(final_shape)
        return _op.reshape(output, final_shape)

    if a_rank == 1:
        # TODO(vvchernov): There should be qnn.matmul but it is not implemented
        # return _op.squeeze(_qnn.op.matmul(_op.expand_dims(a, axis=0),
        #                                   b,
        #                                   a_zp_scalar,
        #                                   b_zp_scalar,
        #                                   a_scale_scalar,
        #                                   b_scale_scalar,
        #                                   transform_num_hidden_units,
        #                                   matmul_result_dtype,
        #                                  ),
        #                    axis=[0]
        #                   )
        if isinstance(a, _expr.Constant):
            a_expand = _expr.const(np.expand_dims(a.data.numpy(), axis=0), dtype=a.data.dtype)
        else:
            a_expand = _op.expand_dims(a, axis=0)
        if isinstance(b, _expr.Constant):
            bt = _expr.const(np.transpose(b.data.numpy(), axes=(1, 0)), dtype=b.data.dtype)
        else:
            bt = _op.transpose(b)

        return _op.squeeze(
            _qnn.op.dense(
                a_expand,
                bt,
                a_zp_scalar,
                b_zp_scalar,
                a_scale_scalar,
                b_scale_scalar,
                transform_num_hidden_units,
                matmul_result_dtype,
            ),
            axis=[0],
        )

    # Otherwise a simple dense op will get the job done.
    if isinstance(b, _expr.Constant):
        bt = _expr.const(np.transpose(b.data.numpy(), axes=(1, 0)), dtype=b.data.dtype)
    else:
        bt = _op.transpose(b)
    return _qnn.op.dense(
        a,
        bt,
        a_zp_scalar,
        b_zp_scalar,
        a_scale_scalar,
        b_scale_scalar,
        transform_num_hidden_units,
        matmul_result_dtype,
    )


def ensure_scalar_shape(x):
    """
    Assume that `x` is a tensor with one element (regardless of tensor rank).
    Return a version of that tensor with rank 0.
    """
    x_shape = infer_shape(x)
    x_rank = len(x_shape)

    if x_rank == 0:
        return x

    num_elem = np.prod(x_shape)
    assert num_elem == 1, f"Cannot squeeze tensor shape {x_shape} to scalar form."

    if isinstance(x, _expr.Constant):
        out = _expr.const(np.squeeze(x.data.numpy()), x.data.dtype)
    else:
        out = _op.squeeze(x)
    return out


class QLinearMatMulOpt(QLinearMatMul):
    """
    Operator converter for QLinearMatMul from Microsoft onnxruntime contrib opset.

    Limitations:
    - Not guaranteed to meet the integer-overflow behavior stipulated in the
      ONNX documentation for this operator.

    The QLinearMatMul converter is re-used for MatMulInteger and is adapted for
    the latter with the optional `expected_out_dtypes` argument.
    """

    @classmethod
    def _impl_v10(cls, inputs, attr, params, expected_out_dtypes=None):
        if expected_out_dtypes is None:
            # The default QLinearMatMul converter is expected to have one of
            # these output dtypes.
            expected_out_dtypes = ["int8", "uint8"]

        # Some of the ops used below take scalar-like inputs, and may require either
        # of the following:
        #
        # - the input is Const node (not merely an expression that *could* be reduced
        #   to a single Const at graph-compilation time)
        #
        # - the input has a specific dtype
        #
        # This function attempts to present 'x' in a form that meets both of those
        # requirements.
        def try_resolve_to_const(x, dtype_override=None):
            x2 = try_resolve_var_to_const(x, params)
            num_elem = np.prod(infer_shape(x))
            if num_elem == 1:
                x2 = ensure_scalar_shape(x2)
            x_dtype = infer_type(x).checked_type.dtype
            if (dtype_override is not None) and (dtype_override != x_dtype):
                if isinstance(x2, _expr.Constant):
                    x2 = _expr.const(x2.data.numpy().astype(dtype_override), dtype_override)
                else:
                    x2 = _op.cast(x2, dtype_override)
            x3 = fold_constant(x2)
            return x3

        # Unpack the inputs and obtain some type info...
        a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp = inputs
        check_quantize_constraint(zp=a_zp, id_name="QLinearMatMul_a")
        check_quantize_constraint(zp=b_zp, id_name="QLinearMatMul_b")
        check_quantize_constraint(zp=y_zp, id_name="QLinearMatMul_y")

        a_type = infer_type(a).checked_type  # 'T1' in ONNX doc for this op
        a_scale_type = infer_type(a_scale).checked_type
        a_zp_type = infer_type(a_zp).checked_type

        b_type = infer_type(b).checked_type  # 'T2' in ONNX doc for this op
        b_scale_type = infer_type(b_scale).checked_type
        b_zp_type = infer_type(b_zp).checked_type

        y_scale_type = infer_type(y_scale).checked_type
        y_zp_type = infer_type(y_zp).checked_type  # 'T3' in ONNX doc for this op

        # Verify type assumptions, based on the ONNX doc for this op...
        assert a_type.dtype in ["int8", "uint8"]
        assert a_scale_type.dtype == "float32"
        assert a_zp_type.dtype == a_type.dtype

        assert b_type.dtype in ["int8", "uint8"]
        assert b_scale_type.dtype == "float32"
        assert b_zp_type.dtype == b_type.dtype

        assert y_scale_type.dtype == "float32"
        assert y_zp_type.dtype in expected_out_dtypes

        a_scale_shape = get_const_tuple(a_scale_type.shape)
        a_zp_shape = get_const_tuple(a_zp_type.shape)
        b_scale_shape = get_const_tuple(b_scale_type.shape)
        b_zp_shape = get_const_tuple(b_zp_type.shape)
        assert (
            a_scale_shape == a_zp_shape
        ), f"Scale and zero point of a must have same shape, but {a_scale_shape} vs {a_zp_shape}"
        assert (
            b_scale_shape == b_zp_shape
        ), f"Scale and zero point of b must have same shape, but {b_scale_shape} vs {b_zp_shape}"
        assert (
            len(a_scale_shape) <= 1 and len(b_scale_shape) <= 1
        ), "dim of a_scale and b_scale must be less or equal to 1"

        # _qnn.op.dense requires the zero-point values to have dtype int32.
        a_scale_scalar = try_resolve_to_const(a_scale)
        a_zp_scalar = try_resolve_to_const(a_zp, "int32")

        b_scale_scalar = try_resolve_to_const(b_scale)
        b_zp_scalar = try_resolve_to_const(b_zp, "int32")

        y_scale_scalar = try_resolve_to_const(y_scale)
        y_zp_scalar = try_resolve_to_const(y_zp, "int32")

        # TODO: Confirm that we're using 'num_hidden_units' correctly / as intended with
        # the '_qnn.op.dense' instance below.
        # num_hidden_units = infer_shape(b)[-1]
        if not _ty.is_dynamic(a_type):
            ishape = list(get_const_tuple(a_type.shape))
            a_shape = _expr.const(ishape, "int64")
            a_rank = len(ishape)
        else:
            a_shape = _op.shape_of(a, "int64")
            a_rank = len(a_type.shape)
        if not _ty.is_dynamic(b_type):
            ishape = list(get_const_tuple(b_type.shape))
            b_shape = _expr.const(ishape, "int64")
            b_rank = len(ishape)
            num_hidden_units = ishape[-1]
        else:
            b_shape = _op.shape_of(b, "int64")
            b_rank = len(b_type.shape)
            num_hidden_units = list(get_const_tuple(b_type.shape))[-1]

        # - Specify the matmul result dtype as int32, so that hopefully the matmul will use
        #   a 32-bit accumulator as seems to be required by the ONNX op's documentation.
        #
        # TL;DR:
        # The ONNX documentation for this op is clear about acceptable overflow
        # behavior during the matmul operation:
        #   - The scalar multiplication ops MAY NOT overflow.
        #   - The scalar addition ops, which sum the results of the scalar multiplication,
        #     MAY overflow, but if they do so, it must behave as one would expect during
        #     32-bit integer-addition overflow.
        # As of this writing, Relay's qnn.op.dense operator doesn't expose a way for us to
        # express these constraints.
        #
        # TODO: Extend TVM / Relay / TIR / etc. to allow this kind of constraint to be
        # expressed in a Relay graph. And then update this importer and various TVM
        # backends accordingly.
        matmul_result_dtype = "int32"
        # TODO(vvchernov): possibly it is better to use unsigned type for result
        # if input types are unsigned:
        # if a_type.dtype == "uint8" and b_type.dtype == "uint8":
        #     matmul_result_dtype = "uint32"

        matmul_result = qmatmul(
            a,
            b,
            a_zp_scalar,
            b_zp_scalar,
            a_scale_scalar,
            b_scale_scalar,
            num_hidden_units,
            matmul_result_dtype,
            a_shape,
            a_rank,
            b_shape,
            b_rank,
            b_type,
        )

        # This information might only be found in the C++ code-comments for the
        # dense.matmul op, but the quantized tensor returned by _qnn.op.dense
        # has scale==(a_scale_scalar * b_scale_scalar), and zero_point==0.
        #
        # 'matmul_result_zp_scalar' has type 'int32' to satisfy input requirements
        # of the [de/re]quantize ops below.
        if isinstance(a_scale_scalar, _expr.Constant) and isinstance(
            b_scale_scalar, _expr.Constant
        ):
            matmul_result_scale_scalar = _expr.const(
                np.multiply(a_scale_scalar.data.numpy(), b_scale_scalar.data.numpy()),
                a_scale_scalar.data.dtype,
            )
        else:
            matmul_result_scale_scalar = fold_constant(_op.multiply(a_scale_scalar, b_scale_scalar))
        matmul_result_zp_scalar = _op.const(0, dtype="int32")

        if "int32" in expected_out_dtypes:
            # This is the adaptation of the QLinearMatMul converter for MatMulInteger,
            # in the MatMulInteger case we skip the unnecessary requantization step.
            return matmul_result

        # requantize requires y_scale to be constant,
        # if y_scale is not constant, doing dequantize -> quantize
        if isinstance(y_scale_scalar, _expr.Constant):
            y = _qnn.op.requantize(
                matmul_result,
                matmul_result_scale_scalar,
                matmul_result_zp_scalar,
                y_scale_scalar,
                y_zp_scalar,
                axis=-1,
                rounding="TONEAREST",
                out_dtype=y_zp_type.dtype,
            )
        else:
            matmul_result_deq = _qnn.op.dequantize(
                matmul_result, matmul_result_scale_scalar, matmul_result_zp_scalar, axis=0
            )

            y = _qnn.op.quantize(
                matmul_result_deq, y_scale_scalar, y_zp_scalar, axis=0, out_dtype=y_zp_type.dtype
            )

        return y


class QLinearAveragePoolOpt(QLinearAveragePool):
    """Operator converter for QLinearAveragePool from Microsoft onnxruntime contrib opset."""

    name = "avg_pool"

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        x_scale = get_scalar(inputs[1], params)
        check_quantize_constraint(zp=inputs[2], id_name="QLinearAveragePool_x")
        x_zero_point = get_scalar(inputs[2], params, dtype="int32")
        y_scale = fold_constant(get_scalar(inputs[3], params))
        check_quantize_constraint(zp=inputs[4], id_name="QLinearAveragePool_y")
        y_zero_point = get_scalar(inputs[4], params, dtype="int32")
        use_float = attr.get("use_float", False)
        if "use_float" in attr:
            del attr["use_float"]

        attr_cvt, data = cls._run_calculation(inputs, attr, params)
        out = attr_cvt([data], attr, params)
        pad = [i.value for i in out.attrs.padding]
        # Onnxruntime doesn't actually do this op in integer, they dequantize to fp32
        # and then requantize afer (according to documentation below)
        # https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.QLinearAveragePool
        # input_dtype = infer_type(data).checked_type.dtype

        x_zp_type = infer_type(inputs[2]).checked_type.dtype
        if use_float or (not attr.get("count_include_pad", False) and any(pad)):
            float_node = _qnn.op.dequantize(data, x_scale, x_zero_point, axis=1)
            out = attr_cvt([float_node], attr, params)
            return _qnn.op.quantize(out, y_scale, y_zero_point, axis=1, out_dtype=x_zp_type)

        out = attr_cvt([data], attr, params)
        out = _qnn.op.requantize(
            out,
            x_scale,
            x_zero_point,
            y_scale,
            y_zero_point,
            axis=1,
            rounding="TONEAREST",
            out_dtype=x_zp_type,
        )
        return out


class QLinearGlobalAveragePoolOpt(QLinearGlobalAveragePool):
    "Operator converter for QLinearGlobalAveragePool from Microsoft onnxruntime contrib opset."

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        input_infer = infer_type(inputs[0]).checked_type
        input_dtype = input_infer.dtype
        rank = len(input_infer.shape)

        x_scale = get_scalar(inputs[1], params)
        check_quantize_constraint(zp=inputs[2], id_name="QLinearGlobalAveragePool_x")
        x_zero_point = get_scalar(inputs[2], params, dtype="int32")
        y_scale = fold_constant(get_scalar(inputs[3], params))
        check_quantize_constraint(zp=inputs[4], id_name="QLinearGlobalAveragePool_y")
        y_zero_point = get_scalar(inputs[4], params, dtype="int32")

        def global_avg_pool(x, rank):
            if rank == 3:
                out = _op.nn.global_avg_pool1d(x)
            elif rank == 4:
                out = _op.nn.global_avg_pool2d(x)
            elif rank == 5:
                out = _op.nn.global_avg_pool3d(x)
            else:
                raise NotImplementedError(
                    "Global avg_pooling is only implemented for 1D, 2D, and 3D kernels, got %dD."
                    % (rank - 2)
                )
            return out

        # Onnxruntime documentation does not mention that this global avg_pool should follow the
        # sequence dequantize -> float op -> quantize, but that is how QLinearAveragePool is done.
        #
        # This op also follows the same pattern since qnn op is not available right now.
        # TODO: Generate QNN op to perform quantized operation instead of dequant -> op -> quant
        if attr.get("use_float", None):
            x = _qnn.op.dequantize(inputs[0], x_scale, x_zero_point, axis=1)
            out = global_avg_pool(x, rank)
            return _qnn.op.quantize(out, y_scale, y_zero_point, axis=1, out_dtype=input_dtype)
        out = global_avg_pool(inputs[0], rank)
        out = _qnn.op.requantize(
            out,
            x_scale,
            x_zero_point,
            y_scale,
            y_zero_point,
            axis=1,
            rounding="TONEAREST",
            out_dtype=input_dtype,
        )
        return out


def get_custom_op(op_name):
    try:
        op = make_custom_op(op_name)
    except AttributeError:
        op = None
    if not op:
        raise tvm.error.OpNotImplemented("Unable to map op_name {} to relay".format(op_name))
    return op


def custom_convert_map(opset):
    """custom op convert"""
    return {
        "static_nms": StaticNms.get_converter(opset),
        "static_batched_nms": StaticBatchedNms.get_converter(opset),
    }


def rewrite_convert_map(opset):
    """rewrite_convert_map
    It is specific to edgex or other reasons(e.g. extended op)
    """
    return {
        "Dropout": Dropout.get_converter(opset),
        "Clip": Clip.get_converter(opset),
        "Conv": ConvOpt.get_converter(opset),
        "BatchNormalization": BatchNormOpt.get_converter(opset),
        "InstanceNormalization": InstanceNormOpt.get_converter(opset),
        "MaxPool": MaxPoolOpt.get_converter(opset),
        "AveragePool": AveragePoolOpt.get_converter(opset),
        "MatMul": MatMulOpt.get_converter(opset),
        "Shape": ShapeOpt.get_converter(opset),
        "Squeeze": SqueezeOpt.get_converter(opset),
        "Pow": PowOpt.get_converter(opset),
        "Softmax": SoftmaxOpt.get_converter(opset),
        "Expand": ExpandOpt.get_converter(opset),
        "Unsqueeze": UnsqueezeOpt.get_converter(opset),
        "Slice": SliceOpt.get_converter(opset),
        "Gemm": GemmOpt.get_converter(opset),
        "ReduceL2": ReduceL2Opt.get_converter(opset),
        "ReduceMax": ReduceMaxOpt.get_converter(opset),
        "ReduceMin": ReduceMinOpt.get_converter(opset),
        "ReduceSum": ReduceSumOpt.get_converter(opset),
        "ReduceMean": ReduceMeanOpt.get_converter(opset),
        "ReduceProd": ReduceProdOpt.get_converter(opset),
        "ReduceLogSumExp": ReduceLogSumExpOpt.get_converter(opset),
        "Concat": ConcatOpt.get_converter(opset),
        "Reshape": ReshapeOpt.get_converter(opset),
        "Transpose": TransposeOpt.get_converter(opset),
        "Relu": ReluOpt.get_converter(opset),
        "LeakyRelu": LeakyReluOpt.get_converter(opset),
        "Sqrt": SqrtOpt.get_converter(opset),
        "Exp": ExpOpt.get_converter(opset),
        "Log": LogOpt.get_converter(opset),
        "Sigmoid": SigmoidOpt.get_converter(opset),
        "Cos": CosOpt.get_converter(opset),
        "Sin": SinOpt.get_converter(opset),
        "Gather": GatherOpt.get_converter(opset),
        "GatherElements": GatherElementsOpt.get_converter(opset),
        "Add": AddOpt.get_converter(opset),
        "Sub": SubOpt.get_converter(opset),
        "Mul": MulOpt.get_converter(opset),
        "Div": DivOpt.get_converter(opset),
        "And": AndOpt.get_converter(opset),
        "Or": OrOpt.get_converter(opset),
        "Not": NotOpt.get_converter(opset),
        "Equal": EqualOpt.get_converter(opset),
        "Where": WhereOpt.get_converter(opset),
        "Tile": TileOpt.get_converter(opset),
        "Round": RoundOpt.get_converter(opset),
        "Reciprocal": ReciprocalOpt.get_converter(opset),
        "Range": RangeOpt.get_converter(opset),
        "Cast": CastOpt.get_converter(opset),
        "Identity": Identity.get_converter(opset),
        "GlobalAveragePool": GlobalAveragePoolOpt.get_converter(opset),
        "GlobalMaxPool": GlobalMaxPoolOpt.get_converter(opset),
        "Pad": PadOpt.get_converter(opset),
        "Upsample": UpsampleOpt.get_converter(opset),
        "ConstantOfShape": ConstantOfShapeOpt.get_converter(opset),
        "Split": SplitOpt.get_converter(opset),
        "ScatterElements": ScatterElementsOpt.get_converter(opset),
        "ScatterND": ScatterNDOpt.get_converter(opset),
        "Resize": ResizeOpt.get_converter(opset),
        "LayerNormalization": LayerNormalizationOpt.get_converter(opset),
        "GroupNormalization": GroupNormalization.get_converter(opset),
        "Flatten": FlattenOpt.get_converter(opset),
        "QuantizeLinear": QuantizeLinearOpt.get_converter(opset),
        "DequantizeLinear": DequantizeLinearOpt.get_converter(opset),
        "QLinearAdd": QLinearAddOpt.get_converter(opset),
        "QLinearMul": QLinearMulOpt.get_converter(opset),
        "QLinearConv": QLinearConvOpt.get_converter(opset),
        "QGemm": QGemmOpt.get_converter(opset),
        "QLinearConcat": QLinearConcatOpt.get_converter(opset),
        "QLinearMatMul": QLinearMatMulOpt.get_converter(opset),
        "QLinearSigmoid": QLinearSigmoidOpt.get_converter(opset),
        "QLinearSoftmax": QLinearSoftmaxOpt.get_converter(opset),
        "QLinearAveragePool": QLinearAveragePoolOpt.get_converter(opset),
        "QLinearGlobalAveragePool": QLinearGlobalAveragePoolOpt.get_converter(opset),
        "QLinearLeakyRelu": QLinearLeakyReluOpt.get_converter(opset),
    }


def Timer(func):
    """timer"""
    timer_dict = {}
    hit = {}

    def wrapper(*args, **kwargs):
        nonlocal timer_dict
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        interval = end - start
        op_name = args[1]
        if op_name in timer_dict:
            timer_dict[op_name] += interval
            hit[op_name] += 1
        else:
            timer_dict[op_name] = interval
            hit[op_name] = 1
        # print("[{func}] time: {time:.5f}s".format(func=func.__name__, time=(end - start)))
        timer_order = OrderedDict(sorted(timer_dict.items(), key=lambda x: x[1], reverse=True))
        for key, value in timer_order.items():
            print(f"{key: <18}{value: 8.6f}{hit[key]: 8d}")
        return result

    return wrapper


class ExtendGraphProto(GraphProto):
    """extend GraphProto for custom op"""

    def __init__(
        self,
        shape,
        dtype,
        freeze_params=False,
        op_type_dict=None,
        is_qir=False,
        nodes_input_occu_num=None,
        nodes_input_shape=None,
        nodes_input_dtype=None,
        total_ops_num=None,
    ):
        super().__init__(shape, dtype, freeze_params, op_type_dict)
        self.convert_map = {}
        self.is_qir = is_qir
        self.custom_ops_dict = {}
        self.nodes_input_occu_num = nodes_input_occu_num
        self.nodes_input_shape = nodes_input_shape
        self.nodes_input_dtype = nodes_input_dtype
        self.dyn_nodes_op = []
        self.total_ops_num = total_ops_num
        self.op_cnt = 0
        # exclude: "QLinearSigmoid" and "QLinearSoftmax" will not be supported!
        self.qnn_op_use_float = [
            "QLinearAdd",
            "QLinearAveragePool",
            "QLinearGlobalAveragePool",
            "QLinearLeakyRelu",
        ]

    def _creat_convert_map(self):
        """creat convert map for extending ops"""
        self.convert_map = dict(_get_convert_map(self.opset), **custom_convert_map(self.opset))
        self.convert_map = dict(self.convert_map, **rewrite_convert_map(self.opset))
        if self.is_qir:
            self.convert_map = dict(self.convert_map, **qir_convert_map(self.opset))

    def _check_for_unsupported_ops(self, graph):
        """check for unsupported ops"""
        self._creat_convert_map()  # creat op map!
        convert_map = self.convert_map
        unsupported_ops = set()
        temp_supported_ops = set()
        for node in graph.node:
            op_name = node.op_type
            if (
                op_name not in convert_map
                and op_name != "Constant"
                and op_name not in _identity_list
                and not check_inside_custom_ops_list(op_name.upper())
            ):
                unsupported_ops.add(op_name)

            if op_name in ["QLinearSoftmax"]:  # remove "QLinearSigmoid" for fixed LUT
                temp_supported_ops.add(op_name)

        if unsupported_ops:
            msg = "The following operators are not supported for frontend ONNX: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)
        if temp_supported_ops:
            msg = f"convert {temp_supported_ops} to op with float. It will not be supported!"
            warnings.warn(msg)

    def _get_call_tir_op(self, op_name, inputs, attrs):
        _inputs = []
        TensorType_inputs = []
        index_list = []
        for index, _input in enumerate(inputs):
            index_list.append(index)
            temp_dict = {}
            if isinstance(_input, tvm.relay.expr.Var):
                in_shape = tuple(shape.value for shape in _input.type_annotation.shape)
                temp_dict = {
                    "shape": in_shape,
                    "dtype": _input.type_annotation.dtype,
                    "value": None,
                    "index": index,
                }
                _inputs.append(temp_dict)
                TensorType_inputs.append(relay.TensorType(in_shape, _input.type_annotation.dtype))
            elif isinstance(_input, tvm.relay.expr.Call):
                func = relay.frontend.common.infer_type(_input)
                in_shape = tuple(d.value for d in func._checked_type_.shape)
                temp_dict = {
                    "shape": in_shape,
                    "dtype": func._checked_type_.dtype,
                    "value": None,
                    "index": index,
                }
                _inputs.append(temp_dict)
                TensorType_inputs.append(relay.TensorType(in_shape, func._checked_type_.dtype))
            elif isinstance(_input, tvm.relay.expr.Constant):
                temp_dict = {
                    "shape": _input.data.shape,
                    "dtype": _input.data.dtype,
                    "value": _input,
                    "index": index,
                }
                _inputs.append(temp_dict)
                TensorType_inputs.append(relay.TensorType(_input.data.shape, _input.data.dtype))
            else:
                raise TypeError("Operator {}'s type is not supported.".format(op_name))
        # out shape infer
        outputs = get_custom_op(op_name.upper()).infer_type(_inputs, attrs)
        assert isinstance(outputs, list), "Please return a list for infer_type."
        _primfunc = get_custom_op(op_name.upper()).gen_kernel(_inputs, attrs, outputs)
        # Check if the inputs have changed
        if len(index_list) != len(_inputs):
            index_list = [info["index"] for info in _inputs]

        # Generate call_tir operator
        TensorType_inputs = [TensorType_inputs[i] for i in index_list]
        if len(outputs) == 1:
            _ty = relay.FuncType(
                TensorType_inputs,
                relay.TensorType(outputs[0]["shape"], outputs[0]["dtype"]),
            )
        elif len(outputs) > 1:
            TensorType_outputs = [
                relay.TensorType(output["shape"], output["dtype"]) for output in outputs
            ]
            _ty = relay.FuncType(
                TensorType_inputs,
                relay.TupleType(TensorType_outputs),
            )
        else:
            raise ValueError("Operator {}'s outputs is wrong.".format(op_name))
        _gv = relay.GlobalVar(op_name + "_custom", type_annot=_ty)
        inputs = [inputs[i] for i in index_list]
        self.custom_ops_dict[_gv] = _primfunc
        out = relay.Call(relay.op.op.get("call_tir"), (_gv, relay.Tuple((*inputs,))))
        if len(outputs) > 1:
            out = relay.TupleWrapper(out, len(outputs))
        return out

    # @Timer  # for test!
    def _convert_operator(self, op_name, inputs, attrs, opset):
        """Convert ONNX operator into a Relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of tvm.relay.function.Function
            List of inputs.
        attrs : dict
            Dict of operator attributes
        opset : int
            Opset version

        Returns
        -------
        sym : tvm.relay.function.Function
            Converted relay function
        """
        convert_map = self.convert_map
        if check_inside_custom_ops_list(op_name.upper()):
            sym = self._get_call_tir_op(op_name, inputs, attrs)
        elif op_name in _identity_list:
            sym = get_relay_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            sym = convert_map[op_name](inputs, attrs, self._params)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym

    def _construct_nodes(self, graph, add_nms=False, r_nms=None, qnn_float_ops_cfg=None):
        """Nodes are stored as directed acyclic graph."""
        op_type_for_shape = NodeInfo.op_type_for_shape()
        op_type_for_dtype = NodeInfo.op_type_for_dtype()
        op_type_for_nofold = NodeInfo.op_type_for_nofold()
        # fold_op_name_list = []  # for test
        # fold_op_elapsed_time = {}
        if qnn_float_ops_cfg:
            assert isinstance(
                qnn_float_ops_cfg, dict
            ), "qnn_float_ops_cfg should be a dict when it is not None!"
        if qnn_float_ops_cfg:
            float_ops_set = set(qnn_float_ops_cfg.keys())
            assert float_ops_set.issubset(
                set(self.qnn_op_use_float)
            ), f"configured qnn float {float_ops_set} should be in {set(self.qnn_op_use_float)}!"

        # pre_progress = 0
        for node in graph.node:
            logger.debug(f"[Frontend] parsing: {node.name}")
            op_name = node.op_type
            attr = self._parse_attr(node.attribute)
            # Fill in span of inputs
            node_source_name = get_source_name(node, self._op_type_dict)
            self._set_parameter_span(node, node_source_name)
            # Create and populate input list.
            inputs = onnx_input()
            for i in node.input:
                if i != "":
                    inputs.append(self._nodes[self._renames.get(i, i)])
                else:
                    inputs.append(None)
            if op_name in op_type_for_shape:
                attr["inferred_shape"] = []
                for i in node.input:
                    if i in self.nodes_input_shape:
                        attr["inferred_shape"].append(self.nodes_input_shape[i])
                    else:
                        attr["inferred_shape"].append(None)
            if op_name in op_type_for_dtype:
                attr["inferred_dtype"] = []
                for i in node.input:
                    if i in self.nodes_input_dtype:
                        attr["inferred_dtype"].append(self.nodes_input_dtype[i])
                    else:
                        attr["inferred_dtype"].append(None)

            if qnn_float_ops_cfg and qnn_float_ops_cfg.get(op_name, None):
                assert (
                    isinstance(qnn_float_ops_cfg[op_name], (list, tuple))
                    or qnn_float_ops_cfg[op_name] == "all"
                ), f"{qnn_float_ops_cfg[op_name]} should be a list/tuple or 'all'!"
                if qnn_float_ops_cfg[op_name] == "all":
                    attr["use_float"] = True
                else:
                    if node_source_name in qnn_float_ops_cfg[op_name]:
                        attr["use_float"] = True

            i_name = self._parse_value_proto(node)
            node_output = self._fix_outputs(op_name, node.output)
            attr["tvm_custom"] = {}
            attr["tvm_custom"]["name"] = i_name
            attr["tvm_custom"]["num_outputs"] = len(node_output)

            if add_nms and op_name == "NonMaxSuppression":
                r_nms.get_exist_nms_info(inputs, attr, self.opset)

            op = self._convert_operator(op_name, inputs, attr, self.opset)
            if not isinstance(op, _expr.TupleWrapper):
                outputs_num = 1
            else:
                outputs_num = len(op)

            if not (self.is_qir or op_name in op_type_for_nofold):
                # t_s = time.time()
                if outputs_num == 1:
                    op = fold_constant(op)
                else:
                    op = _expr.TupleWrapper(fold_constant(op.astuple()), len(op))
                # t_e = time.time()
                # if op_name in fold_op_elapsed_time:
                #     fold_op_elapsed_time[op_name] += t_e - t_s
                # else:
                #     fold_op_elapsed_time[op_name] = t_e - t_s
                # logging.debug(f"{fold_op_elapsed_time}")
                # fold_op_name_list.append(op_name)  # for test!
            else:
                if outputs_num != 1:
                    op = _expr.TupleWrapper(op.astuple(), len(op))

            # TODO: when a onnx op is mapped to several relay ops, the span names of the relay ops
            # will be the same! It's confused for layer comparision!
            op = set_span(op, node_source_name)

            if outputs_num > 1:
                # ONNX supports optional outputs for some nodes.
                # This block searches for missing outputs in the ONNX graph
                # and removes any unneeded ops
                valid_outputs = [False] * outputs_num
                for i, output in enumerate(node_output):
                    if output != "":
                        valid_outputs[i] = True
                # If we have outputs ONNX isn't expecting, we need to drop them
                if not all(valid_outputs):
                    tup = op.astuple()
                    # TupleWrapper can also wrap ops with TupleType outputs
                    if isinstance(tup, _expr.Tuple):
                        # For tuples, we extract the fields instead of using GetTupleItem
                        outputs = [tup.fields[i] for i, valid in enumerate(valid_outputs) if valid]
                    else:
                        # For call nodes, we need to GetTupleItem
                        outputs = [op[i] for i, valid in enumerate(valid_outputs) if valid]
                    # Create the new op with valid outputs
                    if len(outputs) == 1:
                        op = outputs[0]
                    elif len(outputs) != outputs_num:
                        op = _expr.TupleWrapper(_expr.Tuple(outputs), len(outputs))
                    # Drop invalid outputs for the onnx node
                    outputs_num = len(outputs)
                    node_output = [output for output in node_output if output != ""]

            # is_update for split number
            if not (op_name == "Split" and attr.get("is_update", False)):
                assert (
                    len(node_output) == outputs_num
                ), "Number of output mismatch {} vs {} in {}.".format(
                    len(node_output), outputs_num, op_name
                )

            if outputs_num == 1:
                self._nodes[node_output[0]] = op
            else:
                tv = op.tuple_value  # op.astuple()
                if isinstance(tv, _expr.Call):
                    const_cond = [False] * outputs_num
                else:
                    const_cond = [isinstance(i, _expr.Constant) for i in tv]
                for k, i in zip(list(node_output), range(len(node_output))):
                    if const_cond[i]:
                        self._nodes[k] = tv[i]  # simplify TupleGetItemNode(Tuple(*), i)
                    else:
                        self._nodes[k] = op[i]

            # TODO: A node from framework will generate one or more relay op. We can't determine
            # whether generate a dyn op or not from the the last op! Just a rough judgment!!
            # We can config some op name which will generate a dyn op in the list beforehand.
            if isinstance(op, tvm.relay.expr.Call):
                if hasattr(op, "op") and hasattr(op.op, "name") and "dyn" in op.op.name:
                    self.dyn_nodes_op.append(node.name)
            elif isinstance(op, tvm.relay.expr.TupleWrapper):
                tv = op.tuple_value
                if hasattr(tv, "op") and hasattr(tv.op, "name") and "dyn" in tv.op.name:
                    self.dyn_nodes_op.append(node.name)

            # TODO：mask progress bar function to present in unified form in the future
            # if logging.getLevelName(logger.level) == "I":
            #     self.op_cnt += 1
            #     ops_percent = self.op_cnt * 100 // self.total_ops_num
            #     show_progress = ops_percent // 2
            #     if show_progress > pre_progress:
            #         print(
            #             f"\rprogress: {ops_percent:>2d}%", "▋" * show_progress, end="", flush=True
            #         )
            #     pre_progress = show_progress

        # if logging.getLevelName(logger.level) == "I":
        #     print()

        # print(dict(Counter(fold_op_name_list)))  # for test
        # print(fold_op_elapsed_time)

    def from_onnx(
        self,
        graph,
        opset,
        get_output_expr=False,
        out_names=None,
        add_nms=False,
        nms_info_config=None,
        qnn_float_ops_cfg=None,
    ):
        """Construct Relay expression from ONNX graph.

        Onnx graph is a python protobuf object.
        The companion parameters will be handled automatically.
        However, the input names from onnx graph is vague, mixing inputs and
        network weights/bias such as "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph

        opset : opset version

        get_output_expr: bool
            If set to true, this conversion will return each output expression rather
            than a packaged module. This can be useful when converting subgraphs to
            relay.

        out_names: list or tuple
            use for config output name

        add_nms: bool
            whether to add nms after out_name ops

        nms_info_config: dict
            config for add_nms

        qnn_float_ops_cfg: dict
            config for qnn

        Returns
        -------
        mod : tvm.IRModule
            The returned relay module

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        self.opset = opset
        self._parse_graph_initializers(graph)
        self._parse_graph_input(graph)
        self._check_user_inputs_in_outermost_graph_scope()
        self._check_for_unsupported_ops(graph)
        r_nms = None
        if add_nms:
            from .utils import RegisterNms

            r_nms = RegisterNms(nms_info_config)
        self._construct_nodes(graph, add_nms, r_nms, qnn_float_ops_cfg)

        # now return the outputs
        if not out_names:
            outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
        else:
            outputs = []
            for out_name in out_names:
                node_output = self._parse_value_proto(out_name)
                assert (
                    node_output in self._nodes
                ), "configured out_name must be node output name, but {} is not!".format(
                    node_output
                )
                outputs.append(self._nodes[node_output])

        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        if add_nms:
            if isinstance(outputs, _expr.Call):
                out_nodes = [outputs]
            else:
                assert isinstance(outputs, _expr.Tuple)
                out_nodes = [field for field in outputs.fields]
            outputs = r_nms.add_nms(nms_input_node=out_nodes)
            if isinstance(outputs, _expr.TupleWrapper):
                outputs = _expr.Tuple([outputs[i] for i, _ in enumerate(outputs)])

        # If requested, directly return the converted expressions.
        if get_output_expr:
            return outputs
        ## Maintain the order of inputs and parameters from the ONNX graph, but only include
        ## those parameters that are needed to execute the relay graph
        free_vars = analysis.free_vars(outputs)
        nodes = {v: k for k, v in self._nodes.items()}
        free_vars = [nodes[var] for var in free_vars]
        for i_name in self._params:
            if i_name in free_vars and i_name not in self._inputs:
                self._inputs[i_name] = self._nodes[i_name]
        # Create a function from our output expression and all input variables.
        func = _function.Function([v for k, v in self._inputs.items()], outputs)
        mod = IRModule.from_expr(func)
        for key, value in self.custom_ops_dict.items():
            mod[key] = value
        # r_nms.save(mod, self._params, "/workspace/")
        return mod, self._params


def get_ops_type(model):
    """get ops type"""
    op_type_list = []
    for node in model.graph.node:
        op_type_list.append(node.op_type)
    return op_type_list


class NodeInfo:
    """A helper class for handling node info from onnx model

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0

    """

    def __init__(self, use_onnx_infer=True):
        self.use_onnx_infer = use_onnx_infer  # onnx or onnxruntime infer

    @classmethod
    def op_type_for_shape(cls):
        """config op_type for getting input shape"""
        op_type_list = [
            "Conv",
            "Shape",
            "Squeeze",
            "Softmax",
            "Expand",
            "Slice",
            "ReduceL2",
            "ReduceMax",
            "ReduceMin",
            "ReduceSum",
            "ReduceMean",
            "ReduceProd",
            "ReduceLogSumExp",
            "Gather",
            "GatherElements",
            "MatMul",
            "Gemm",
            "AveragePool",
            "MaxPool",
            "Unsqueeze",
            "GlobalAveragePool",
            "GlobalMaxPool",
            "Upsample",
            "ScatterElements",
            "ScatterND",
            "Resize",
            "Split",
            "LayerNormalization",
            "GroupNormalization",
            "Flatten",
        ]
        return op_type_list

    @classmethod
    def op_type_for_dtype(cls):
        """config op_type for getting input type"""
        op_type_list = [
            "AveragePool",
            "MaxPool",
            "MatMul",
            "Pow",
            "Gemm",
            "Reciprocal",
            "Range",
            "ScatterElements",
            "ScatterND",
            "GatherElements",
            "LayerNormalization",
            "GroupNormalization",
        ]
        return op_type_list

    @classmethod
    def op_type_for_nofold(cls):
        """config op_type for nofold
        e.g ref swinv2 and others
        """
        op_type_list = [
            "Identity",
            "Constant",
            "LeakyRelu",
            "AveragePool",
            "Conv",
            "Softmax",
            "GlobalAveragePool",
            "GlobalMaxPool",
            "BatchNormalization",
            "Relu",
            "InstanceNormalization",
            "Dropout",
            "Erf",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Pow",
            "Exp",
            "Log",
            "Sqrt",
            "Sigmoid",
            "Clip",
            "Squeeze",
            "Reshape",
            "Transpose",
            "Gemm",
            "Gather",
            "Concat",
            "ReduceMax",
            "ReduceMin",
            "ReduceSum",
            "ReduceMean",
            "ReduceProd",
            "AveragePool",
            "MaxPool",
            "Unsqueeze",
            "MatMul",
            "Slice",
            "ReduceL2",
            "Split",
            "Sin",
            "Cos",
            "And",
            "Or",
            "Not",
            "Equal",
            "Where",
            "Tile",
            "Round",
            "ConstantOfShape",
            "Cast",
            "Reciprocal",
            "Expand",
            "Range",
            "Resize",
            "LayerNormalization",
            "GroupNormalization",
            "Flatten",
            # Quantization
            "QLinearConv",
            "QGemm",
            "QLinearMatMul",
            "QLinearAdd",
            "QLinearMul",
            "QLinearConcat",
            "QLinearAveragePool",
            "QLinearGlobalAveragePool",
            "QLinearLeakyRelu",
            "QLinearSigmoid",
            "QLinearSoftmax",
            "QuantizeLinear",
            "DequantizeLinear",
        ]
        return op_type_list

    def get_dim_from_tensor_shape_info(self, info):
        """get shape from tensor shape info"""
        tensor_shape = [v.dim_value for v in info.shape.dim]
        # dynamic shape or can't infer shape, e.g [unk__188(dim_param), 64(dim_value), ...]
        # however, the dim of shape will be useful! So mask codes as follows:
        # if 0 in tensor_shape:
        #     tensor_shape = []
        return tensor_shape

    def statistics_op_type(self, model):
        """statistics occurrence number of op types"""
        op_type_list = get_ops_type(model)
        total_ops_num = len(op_type_list)
        op_type_number = dict(Counter(op_type_list))

        logger.debug(f"[Frontend] ops info: total({total_ops_num}) => {op_type_number}")

        return total_ops_num

    def add_const_value_to_input(self, model_input_shape, model):
        """add const value for input dim"""
        if not model_input_shape:
            return model

        for i in model.graph.input:
            if i.name in model_input_shape:
                const_ishape = model_input_shape[i.name]
                proto_dim = i.type.tensor_type.shape.dim
                for j, c_dim in enumerate(const_ishape):
                    p_dim = proto_dim[j]
                    if isinstance(c_dim, int):  # c_dim > 0
                        if hasattr(proto_dim, "dim_value") and p_dim.dim_value != c_dim:
                            raise ValueError(
                                f"Can't set dim val {c_dim} for axis{j} of {i.name}. "
                                f"The dim val of model is {p_dim.dim_value}"
                            )
                        p_dim.dim_value = c_dim
                    else:
                        raise ValueError(
                            f"Only support int for input shape dim, " f"but {const_ishape}"
                        )
        return model

    def onnxruntime_infer_shape(self, model):
        """onnxruntime_infer_shape"""
        from onnxruntime.transformers.shape_infer_helper import SymbolicShapeInferenceHelper

        shape_infer_helper = SymbolicShapeInferenceHelper(model, verbose=0)
        inferred_model = shape_infer_helper.infer_shapes(
            model, auto_merge=True, guess_output_rank=False
        )
        return inferred_model

    def info_infer(
        self,
        model_input_shape,
        model_input_dtype,
        model,
        onnx_file_path,
        inferred_model_save_path=None,
    ):
        """info infer"""
        new_model = self.add_const_value_to_input(model_input_shape, model)
        # TODO:onnx.shape_inference.infer_shapes_path for >2GB models
        # LLM will use this API．Maybe we can use infer_shapes for the large model if
        # ModelProto is used as input (if onnx-1.15.0, but onnx-1.12.0 is not)!
        try:
            # It is advantageous to use onnxruntime infer. for model with dynamic input.
            # But onnxruntime infer. is unrobust!
            if not self.use_onnx_infer:
                try:
                    inferred_model = self.onnxruntime_infer_shape(new_model)
                except Exception as err:
                    warnings.warn("onnxruntime infer shape: " + str(err))
                    self.use_onnx_infer = True
            if self.use_onnx_infer:
                inferred_model = onnx.shape_inference.infer_shapes(new_model)
        except onnx.onnx_cpp2py_export.shape_inference.InferenceError:
            nodes_input_occu_num = {}
            nodes_input_shape = {}
            nodes_input_dtype = {}
            return nodes_input_occu_num, nodes_input_shape, nodes_input_dtype
        except ValueError as e:
            if "Message onnx.ModelProto exceeds maximum protobuf size of 2GB" in str(e):
                try:
                    if not inferred_model_save_path:
                        inferred_model_save_path = onnx_file_path
                    onnx.shape_inference.infer_shapes_path(onnx_file_path, inferred_model_save_path)
                    inferred_model = onnx.load(inferred_model_save_path)
                except Exception as err:  # may OOM! or onnx_file_path is not writable!
                    warnings.warn(str(err))
            else:
                warnings.warn(str(e))
            nodes_input_occu_num = {}
            nodes_input_shape = {}
            nodes_input_dtype = {}
            return nodes_input_occu_num, nodes_input_shape, nodes_input_dtype

        value_info = inferred_model.graph.value_info
        nodes_out_shape = {}
        nodes_out_dtype = {}
        if not model_input_shape or not model_input_dtype:
            for i in model.graph.input:
                i_name, i_shape, i_dtype, _ = get_info(i)
                nodes_out_shape[i_name] = i_shape
                nodes_out_dtype[i_name] = i_dtype
        else:
            for key, value in model_input_shape.items():
                nodes_out_shape.update({key: list(value)})
            if isinstance(model_input_dtype, str):
                for key in model_input_shape.keys():
                    nodes_out_dtype[key] = model_input_dtype
            else:
                assert isinstance(
                    model_input_dtype, dict
                ), f"input dtype of model must be str or dict, but {type(model_input_dtype)}"
                nodes_out_dtype.update(model_input_dtype)

        # notes: Not all nodes can infer their shape, for example, the target shape for reshape
        # op is not const! -> {"node_name": []}
        for out in value_info:
            try:
                # onnx.TensorProto.DataType.items() -> elem_type
                tensor_type = out.type.tensor_type
                nodes_out_dtype[out.name] = get_type(tensor_type.elem_type)
                nodes_out_shape[out.name] = self.get_dim_from_tensor_shape_info(tensor_type)
            except KeyError:  # densenet169_tv_in1k
                tensor_type = out.type.sequence_type.elem_type.tensor_type
                nodes_out_dtype[out.name] = get_type(tensor_type.elem_type)
                nodes_out_shape[out.name] = self.get_dim_from_tensor_shape_info(tensor_type)
            except:
                pass

        op_type_for_shape_list = self.op_type_for_shape()
        op_type_for_dtype_list = self.op_type_for_dtype()
        nodes_in = []
        nodes_input_shape = {}
        nodes_input_dtype = {}
        for node in inferred_model.graph.node:
            op_type = node.op_type
            node_input = node.input
            nodes_in.extend(node_input)

            if op_type in op_type_for_shape_list:
                for inp in node_input:
                    if inp in nodes_out_shape:
                        nodes_input_shape[inp] = nodes_out_shape[inp]

            if op_type in op_type_for_dtype_list:
                for inp in node_input:
                    if inp in nodes_out_dtype:
                        nodes_input_dtype[inp] = nodes_out_dtype[inp]

        nodes_input_occu_num = dict(Counter(nodes_in))

        return nodes_input_occu_num, nodes_input_shape, nodes_input_dtype


def from_onnx(
    model,
    shape=None,
    dtype="float32",
    opset=None,
    freeze_params=True,
    convert_config=None,
    export_node_renamed_model_path=None,
    out_names=None,
    is_qir=False,
    add_nms=False,
    nms_info_config=None,
    qnn_config=None,
    onnx_file_path=None,
    external_data_path=None,
    inferred_model_save_path=None,
    use_onnx_infer_shape=True,
):
    """Convert a ONNX model into an equivalent Relay Function.

    ONNX graphs are represented as Python Protobuf objects.
    The companion parameters will be handled automatically.
    However, the input names from onnx graph is vague, mixing inputs and
    network weights/bias such as "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
    retains that dynamism upon import, and the compiler attempts to convert the
    model into a static shapes at compile time. If this fails, there may still
    be dynamic operations in the model. Not all TVM kernels currently support
    dynamic shapes, please file an issue on discuss.tvm.apache.org
    if you hit an error with dynamic kernels.

    example(including custom op):
        from tvm.contrib.edgex.relay.frontend.onnx import from_onnx
        mod, params = from_onnx(model, shape_dict, dtype_dict)

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    opset : int, optional
        Override to autodetected opset.
        This can be helpful for some testing.

    freeze_params: bool
        If this parameter is true, the importer will take any provided
        onnx input values (weights, shapes, etc) and embed them into the relay model
        as Constants instead of variables. This allows more aggressive optimizations
        at compile time and helps in making models static if certain inputs represent
        attributes relay would traditionally consider compile-time constants.
        notes: the default for the parameter is False in nnp200/nnp300, but True in nnp400/edgex.

    convert_config : Optional[Dict[str, Any]]
        Default config:
            use_nt_batch_matmul : bool = True
                True to convert qualified onnx `matmul` to `nn.batch_matmul` strict to NT format
                (transpose_a=False, transpose_b=True).

    export_node_renamed_model_path : str, optional
        Export the node renamed onnx model to the path.
        Some models do not contain names in their nodes. During the conversion, if names of nodes
        are empty, new names will be assigned based on their op types. The exported model can be the
        reference to spans.

    out_names: list or tuple
        use for config output name

    is_qir: bool
        whether to use qir quantization or not

    add_nms: bool
            whether to add nms after out_name ops

    nms_info_config: dict
            config for add_nms

    qnn_config: dict
            config for qnn

    onnx_file_path: str
            path of the *.onnx file

    external_data_path: str
            external data directory for weight > 2GiB

    inferred_model_save_path: str
            used only for large model > 2GiB

    use_onnx_infer_shape: bool
            use onnx or onnxruntime to infer shape

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """
    global ONNX_DEFAULT_CONFIGS
    if convert_config is not None:
        ONNX_DEFAULT_CONFIGS.update(convert_config)

    try:
        if hasattr(onnx.checker, "check_model"):
            # try use onnx's own model checker before converting any model
            try:
                if not external_data_path:
                    onnx.checker.check_model(model)
                else:
                    try:
                        onnx.checker.check_model(onnx_file_path)
                    except RuntimeError:
                        warnings.warn(
                            "check model: onnx file and external data may need to be "
                            "under the same directory!"
                        )
            except Exception as e:  # pylint: disable=c-extension-no-member, broad-except
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(e))
    except ImportError:
        pass

    node_info = NodeInfo(use_onnx_infer_shape)
    total_ops_num = node_info.statistics_op_type(model)  # get op info
    nodes_input_occu_num, nodes_input_shape, nodes_input_dtype = node_info.info_infer(
        shape, dtype, model, onnx_file_path, inferred_model_save_path
    )

    if external_data_path:
        from onnx.external_data_helper import load_external_data_for_model

        load_external_data_for_model(model, base_dir=external_data_path)

    g = ExtendGraphProto(
        shape,
        dtype,
        freeze_params,
        op_type_dict={},
        is_qir=is_qir,
        nodes_input_occu_num=nodes_input_occu_num,
        nodes_input_shape=nodes_input_shape,
        nodes_input_dtype=nodes_input_dtype,
        total_ops_num=total_ops_num,
    )
    graph = model.graph

    try:
        opset_in_model = 1
        if model.opset_import:
            # TODO: for now we only really support ai.onnx op set
            # TODO: handle other namespaces well see https://github.com/apache/tvm/issues/10950
            for opset_identifier in model.opset_import:
                # As per https://github.com/onnx/onnx/blob/main/docs/IR.md
                # All operator sets except the default one must specify the operator version
                if str(opset_identifier.domain) in ["ai.onnx", ""]:
                    opset_in_model = opset_identifier.version
                    break
    except AttributeError:
        opset_in_model = 1

    if opset is None:
        opset = opset_in_model
    elif opset < opset_in_model:
        warnings.warn(
            ""
            f"You are overwritting original opset ver = {opset_in_model} by lower ver = {opset}. "
            f"That might cause model conversion errors."
        )

    qnn_float_ops_cfg = qnn_config.get("float_ops", None) if qnn_config else None
    # Use the graph proto as a scope so that ops can access other nodes if needed.
    with g:
        mod, params = g.from_onnx(
            graph,
            opset,
            out_names=out_names,
            add_nms=add_nms,
            nms_info_config=nms_info_config,
            qnn_float_ops_cfg=qnn_float_ops_cfg,
        )

    if export_node_renamed_model_path:
        export_model(export_node_renamed_model_path, graph)

    if freeze_params and g.dyn_nodes_op and not is_qir:
        mod = relay.transform.DynamicToStatic()(mod)

    return mod, params


def get_model_path(model_path):
    """get model path: onnx file and/or external data"""
    import re

    if not isinstance(model_path, str):
        return model_path

    pattern = r"\((.*?)\)"
    find_str = "("
    match_res = re.findall(pattern, model_path)
    if not match_res:
        pattern = r"\[(.*?)\]"
        find_str = "["
        match_res = re.findall(pattern, model_path)
    if match_res and len(match_res) == 1 and ".onnx" in match_res[0]:
        base_dir = model_path[: model_path.find(find_str)]
        str_list = match_res[0].split(",")
        try:
            path_list = [os.path.join(base_dir, eval(path)) for path in str_list]
        except Exception as e:
            path_list = [os.path.join(base_dir, path.strip()) for path in str_list]

        for path in path_list:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} is not found!")

        return path_list
    else:
        return model_path


def load_model(model_path, shape, **kwargs):
    """convert onnx framework model to relay"""
    model_path = get_model_path(model_path)
    logger.info(f"[Frontend] Loading model: {model_path}")
    onnx_file_path = None
    external_data_path = None

    if isinstance(model_path, str):
        onnx_model = onnx.load(model_path)
        model_size = onnx_model.ByteSize()
        if model_size > 1024**3:  # 1GiB
            warnings.warn(
                f"Attempting to parse a large model({model_size / 1024.0 ** 3: .2f} GiB)."
                " This may require a large amount of memory for running shape infer! "
                "It's good practice to convert the ONNX model to external data, "
                "then config a list/tuple for the model and external data!"
            )
        onnx_file_path = model_path
    elif isinstance(model_path, (tuple, list)):
        # from onnx.external_data_helper import load_external_data_for_model

        assert len(model_path) == 2, f"only need two files, but there are {len(model_path)}"
        onnx_file_path = [file for file in model_path if file.endswith(".onnx")][0]
        path_list = list(model_path)
        path_list.remove(onnx_file_path)
        external_data_path = path_list[0]
        if not os.path.samefile(os.path.dirname(onnx_file_path), external_data_path):
            warnings.warn(
                "It will be more better that onnx model and the external data are "
                "under the same directory!"
            )

        onnx_model = onnx.load(onnx_file_path, load_external_data=False)
        # more slower for next shape infer if loading at first
        # load_external_data_for_model(onnx_model, base_dir=external_data_path)
    else:
        raise TypeError("type of the model_path must be str/tuple/list!")

    if logging.getLevelName(logger.level) == "D":
        initial_op_type_list = get_ops_type(onnx_model)

    try:
        import onnx_graphsurgeon as gs

        graph = gs.import_onnx(onnx_model)
        graph.fold_constants()
        graph.cleanup()
        onnx_model = gs.export_onnx(graph)
        # onnx.save(onnx_model, "xxx.onnx")

        if logging.getLevelName(logger.level) == "D":
            gs_op_type_list = get_ops_type(onnx_model)
            unparsed_ops = set(initial_op_type_list) - set(gs_op_type_list)
            if unparsed_ops:
                logger.debug(f"[Frontend] existent unparsed ops: {unparsed_ops}")
    except ModuleNotFoundError:
        pass

    other_cfgs = {}
    if "extras" in kwargs:
        other_cfgs.update(kwargs["extras"])
    if "dtype" in kwargs:
        other_cfgs["dtype"] = kwargs["dtype"]
    if "qnn_config" in kwargs:
        other_cfgs["qnn_config"] = kwargs["qnn_config"]
    other_cfgs["onnx_file_path"] = onnx_file_path
    other_cfgs["external_data_path"] = external_data_path

    logger.info("[Frontend] Parsing model ...")
    mod, params = from_onnx(model=onnx_model, shape=shape, **other_cfgs)
    logger.info("[Frontend] Parsing success")
    logger.debug(f"[Frontend] after parsing: \n{relay.transform.InferType()(mod)}")

    return mod, params


def from_qir(model, shape_dict, out_names=None):
    """qir importer"""
    dtype_dict = {}
    dtype_map = {
        "FLOAT": "float32",
        "FLOAT16": "float16",
        "UINT8": "uint8",
        "INT8": "int8",
        "UINT16": "uint16",
        "INT16": "int16",
        "INT32": "int32",
        "INT64": "int64",
        "BOOL": "bool",
        "DOUBLE": "float64",
        "UINT32": "uint32",
        "UINT64": "uint64",
    }
    const_name = [i.name for i in model.graph.initializer]
    for i in model.graph.input:
        if i.name not in const_name:
            onnx_type = onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type)
            assert onnx_type in dtype_map, f"Unkown onnx type: {onnx_type}"
            dtype_dict[i.name] = dtype_map[onnx_type]
    mod, params = from_onnx(model, shape_dict, dtype_dict, out_names=out_names, is_qir=True)
    return mod, params
