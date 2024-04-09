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
# pylint: disable=unused-argument,inconsistent-return-statements,bad-continuation,arguments-differ,unused-import,import-outside-toplevel,len-as-condition,too-many-nested-blocks
# pylint: disable=unexpected-keyword-arg,chained-comparison
"""config"""

import logging
import collections
import numpy as np
import tvm
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
from .threshold import Threshold
from .method_dtype import Method, DataType
from .tools import GetFinalConvIdx
from .op_config import OPCONFIGS

LOGGER = logging.getLogger("quantize")


VU_OP_LIST = [
    "mulitply",
    "sum",
    # "nn.pad",
    "transpose",
    "strided_slice",
    "image.resize2d",
    "nn.upsampling",
    "nn.unmaxpool_upsample",
]

# in: int8, out: int8
INT8_OP_LIST = [
    "nn.nnp_conv2d_winograd2d_without_weight_transform_bias_add",
    "conv2d_winograd1d_bias_add",
    "nn.contrib_conv2d_winograd1d_without_weight_transform",
    "nn.nnp_conv2d_winograd2d_without_weight_transform",
    "conv3d_winograd1d_bias_add",
    "nn.contrib_conv3d_winograd1d_without_weight_transform",
]

ELT_WISE_OP_LIST = [
    "add",
]

DTYPE_AS_SAME_AS_INPUT_OP_LIST = [
    "image.resize2d",
]


def helper_get_call_name(call_expr):
    if isinstance(call_expr.op, tvm.relay.Function):
        name = getattr(call_expr.op.attrs, "Composite")
        if not isinstance(name, str):
            name = name.value
    else:
        name = call_expr.op.name
    return name


class ConfigSpace(ExprVisitor):
    """ConfigSpace"""

    def __init__(self, mod, node_id, quantize_config, net_in_dtype, net_in_scale):
        super().__init__()
        self.node_id = node_id
        self.config = collections.OrderedDict()
        self.all_op = []
        self.exp_ref = {}
        self.idx = -1
        self.quantize_config = {} if quantize_config is None else quantize_config
        self.net_in_dtype = net_in_dtype
        self.net_in_scale = net_in_scale
        self.floatlist_use_name = []
        self.floatlist_hit_name = []
        self.first_node = True

        # for span name to set float_list, check the float_list set is ok
        if (
            "float_list" in self.quantize_config
            and len(self.quantize_config["float_list"]) > 0
        ):
            for value in self.quantize_config["float_list"]:
                if isinstance(value, str) and not value.startswith("auto"):
                    self.floatlist_use_name.append(value)
                if isinstance(value, list):
                    for value_mem in value:
                        if isinstance(value_mem, str):
                            self.floatlist_use_name.append(value_mem)
            self.floatlist_use_name = list(set(self.floatlist_use_name))

        # compatible with nnp300
        if not isinstance(mod, tvm.relay.Function):
            self.visit(mod["main"])
        else:
            self.visit(mod)

        # for span name to set float_list, check the float_list set is ok
        self.floatlist_hit_name = list(set(self.floatlist_hit_name))
        for orig_set_name in self.floatlist_use_name:
            if orig_set_name not in self.floatlist_hit_name:
                assert 0, (
                    "set <"
                    + orig_set_name
                    + "> to use float, but the op no-in opt-model, plz check opt_ir.pdf"
                )

        LOGGER.info(f"[config] all op types: {self.all_op}")
        # LOGGER.info()

    @staticmethod
    def get_constant_arg_config(arg_idx, name, quantize_config):
        """get constant config"""
        dtype_tmp = DataType.Int8
        cond1 = name == "nn.bias_add" or (arg_idx == 2 and name.endswith("bias_add"))
        cond2 = (
            name == "nn.bias_add"
            or (arg_idx == 2 and name.endswith("bias_add"))
            and "target" in quantize_config
            and quantize_config["target"].startswith("nnp3")
        )

        if cond1:
            dtype_tmp = DataType.Int32
        if cond2:
            dtype_tmp = DataType.Int24

        arg_dict = {
            "default_config": {
                "threshold": Threshold.MinMax,
                "method": Method.Symmetry,
                "dtype": dtype_tmp,
                "pertensor": False,
            }
        }

        return arg_dict

    @staticmethod
    def get_call_arg_config(
        arg, arg_id, net_in_dtype, config, quantize_config, net_in_scale
    ):
        """get call config"""
        tmp_dict = {
            "default_config": {
                "threshold": Threshold.L2Norm,
                "method": Method.Symmetry,
                "dtype": DataType.Int8,
                "pertensor": False,
            }
        }

        # only support dict
        assert isinstance(net_in_dtype, dict), "net_in_dtype must be dict"
        if isinstance(arg, tvm.relay.Var):
            arg_name = arg.name_hint
            # consider float16 and float32
            if arg_name in net_in_dtype and not net_in_dtype[arg_name].startswith(
                "float"
            ):
                tmp_dict["default_config"]["dtype"] = net_in_dtype[arg_name]
                tmp_dict["default_config"]["is_fixed_var"] = True

        if isinstance(arg, tvm.relay.Var) and isinstance(net_in_scale, dict):
            arg_name = arg.name_hint
            if arg_name in net_in_scale:
                tmp_dict["default_config"]["in_scale"] = net_in_scale[arg_name]

        map_dict = {
            "min_max": Threshold.MinMax,
            "percentile": Threshold.Percentile,
            "l2norm": Threshold.L2Norm,
            "kld": Threshold.KLDAbs,
            "ema": Threshold.MovingAverageMinMax,
            "mse": Threshold.MSE,
        }
        if "calib_method" in quantize_config:
            if quantize_config["calib_method"].startswith("percentile"):
                tmp_dict["default_config"]["threshold"] = quantize_config[
                    "calib_method"
                ]
            else:
                tmp_dict["default_config"]["threshold"] = map_dict[
                    quantize_config["calib_method"]
                ]

        if "l2norm_mp" in quantize_config:
            tmp_dict["default_config"]["l2norm_mp"] = quantize_config["l2norm_mp"]

        if "pertensor" in quantize_config:
            tmp_dict["default_config"]["pertensor"] = quantize_config["pertensor"]

        return tmp_dict

    def get_whether_quantized(self, quantize_config, name, call_id, tmp, call, config):
        """get whether quantized"""

        span_name = "none"
        if call.span is not None:
            if "ir_pass" in tvm.relay.__dict__:
                span_name = call.span.source.name
            else:
                span_name = call.span.source_name.name

        # float list
        cond1 = False
        if "float_list" in quantize_config:
            for value in quantize_config["float_list"]:
                if (isinstance(value, str) and value in [name, span_name]) or (
                    isinstance(value, int) and value == call_id
                ):
                    cond1 = True

                    if isinstance(value, str):
                        self.floatlist_hit_name.append(value)

                elif isinstance(value, list) and (
                    name in value or call_id in value or span_name in value
                ):
                    if name in value:
                        self.floatlist_hit_name.append(name)
                    if span_name in value:
                        self.floatlist_hit_name.append(span_name)
                    cond1 = True
        if cond1:
            config[tmp]["quantized"] = False

        # nnp3xx multiply use fp16
        if (
            "target" in quantize_config
            and quantize_config["target"].startswith("nnp3")
            and name == "multiply"
        ):
            config[tmp]["quantized"] = False

        # equal use fp16
        if name == "equal":
            config[tmp]["quantized"] = False
            for arg in call.args:
                if isinstance(arg, tvm.relay.Call):
                    tmp_arg = self.node_id[arg]
                    config[tmp_arg]["quantized"] = False

        if name in ["add", "subtract", "multiply"]:
            # two args is call, diff shape use fp16
            # arg_1 is const, and can do broadcast can use fix
            cond1 = isinstance(call.args[1], tvm.relay.Constant)
            cond2 = len(call.args[0]._checked_type_.shape) != len(
                call.args[1]._checked_type_.shape
            )
            # cond9 means dynamic shape.
            cond3, cond4 = False, False
            if not cond2:
                arg0_size, arg1_size = 1, 1
                anycls = (
                    tvm.tir.expr.Any
                    if "transform" in tvm.relay.__dict__
                    else tvm.relay.expr.Any
                )
                for i in range(len(call.args[0]._checked_type_.shape)):
                    if isinstance(
                        call.args[0]._checked_type_.shape[i], anycls
                    ) or isinstance(call.args[1]._checked_type_.shape[i], anycls):
                        continue

                    arg0_size = arg0_size * call.args[0]._checked_type_.shape[i].value
                    arg1_size = arg1_size * call.args[1]._checked_type_.shape[i].value
                cond3 = arg0_size != arg1_size
                cond4 = arg0_size < arg1_size
            cond5 = (not cond1 and (cond2 or cond3)) or (cond1 and cond4)

            cond6 = "target" in quantize_config and quantize_config[
                "target"
            ].startswith("nnp3")

            if cond5 and cond6:
                config[tmp]["quantized"] = False

            # subtract arg0 concat use fp16
            cond7 = name == "subtract" and isinstance(call.args[0], tvm.relay.Constant)

            # elem meet const shape < 3 and call shape < 4
            cond8 = (
                name in ["add", "subtract"]
                and isinstance(call.args[1], tvm.relay.Constant)
                and len(call.args[1]._checked_type_.shape) < 3
                and len(call.args[0]._checked_type_.shape) < 4
            )

            if cond7 or cond8:
                config[tmp]["quantized"] = False

        if name == "clip" and (call.attrs.a_min < 0 or call.attrs.a_max > 6):
            config[tmp]["quantized"] = False

        if name == "stack":
            tmp_arg = self.node_id[call.args[0]]
            config[tmp_arg]["quantized"] = False

        # for grid_sample
        def _judge_enough_mem(in_height, in_width, in_dtype="float16"):
            """for 400 vu"""
            grid_vu_seg_size = 256
            index_cn = 4
            weight_cn = 2 if in_dtype in ["int8", "uint8", "float16"] else 4
            out_dtype = in_dtype
            out_cn = 1 if out_dtype in ["int8", "uint8"] else 2
            out_cn = 4 if out_dtype in ["int32", "float32"] else out_cn
            in_cn = 1 if in_dtype in ["int8", "uint8"] else 2
            in_cn = 4 if in_dtype in ["int32", "float32"] else in_cn
            other_size = 2 * out_cn + index_cn + weight_cn * 2
            input_valid_size = 64 * 1024 - other_size * grid_vu_seg_size
            input_channel_size = in_height * in_width * in_cn
            return input_channel_size * 2 <= input_valid_size

        # for 400T
        if (
            name in ["image.grid_sample"]
            and "target" in quantize_config
            and quantize_config["target"].startswith("nnp4")
        ):
            input_shape = call.args[0]._checked_type_.shape
            # not enough for fp16 and enough for int8
            if len(input_shape) == 4 and (
                not _judge_enough_mem(input_shape[-2], input_shape[-1], "float16")
                and _judge_enough_mem(input_shape[-2], input_shape[-1], "int8")
            ):
                pass  # config[tmp]["quantized"] = True

    def get_op_parameter(self, quantize_config, config, tmp, call=None, node_id=None):
        """control op strategy"""
        cond1 = tmp.split("_")[1].startswith("conv2d") and tmp.endswith("bias_add")
        if "bias_method" in quantize_config and cond1:
            config[tmp]["bias_method"] = quantize_config["bias_method"]

        assert (
            "target" in quantize_config
        ), "use relay.quantization.get_quantize_config to get quantize_config"
        config[tmp]["nnp_dev"] = quantize_config["target"]

        map_dict = {
            "min_max": Threshold.MinMax,
            "percentile": Threshold.Percentile,
            "l2norm": Threshold.L2Norm,
            "kld": Threshold.KLDAbs,
            "ema": Threshold.MovingAverageMinMax,
            "mse": Threshold.MSE,
        }

        span_name = "none"
        if call.span is not None:
            if "ir_pass" in tvm.relay.__dict__:
                span_name = call.span.source.name
            else:
                span_name = call.span.source_name.name

        # "op_config"
        tmp_key = None
        # op type
        if (
            "op_config" in quantize_config
            and tmp.strip(tmp.split("_")[0] + "_") in quantize_config["op_config"]
        ):
            tmp_key = tmp.strip(tmp.split("_")[0] + "_")

        # op prefix can overide op type
        if (
            "op_config" in quantize_config
            and tmp.split("_")[0] in quantize_config["op_config"]
        ):
            tmp_key = tmp.split("_")[0]

        # span_name
        if "op_config" in quantize_config and span_name in quantize_config["op_config"]:
            tmp_key = span_name

        if tmp_key is not None:
            op_inputkey = quantize_config["op_config"][tmp_key]
            for k_1, v_1 in op_inputkey.items():
                if isinstance(v_1, dict):
                    for k_2, v_2 in v_1.items():
                        if k_2 == "calib_method":
                            if v_2.startswith("percentile"):
                                config[tmp]["default_config"][k_1]["threshold"] = v_2
                            else:
                                config[tmp]["default_config"][k_1][
                                    "threshold"
                                ] = map_dict[v_2]
                        else:
                            config[tmp]["default_config"][k_1][k_2] = v_2
                else:
                    config[tmp][k_1] = v_1

        # connect concatenate and input-tuple
        if tmp.endswith("concatenate"):
            axis = call.attrs.axis
            if axis < 0:
                axis = len(call._checked_type_.shape) + axis
            arg_tmp = node_id[call.args[0]]
            config[arg_tmp]["axis"] = axis

            if "quantized" in config[tmp]:
                config[arg_tmp]["quantized"] = config[tmp]["quantized"]

        # connect concatenate and input-tuple
        if tmp.endswith("einsum") and isinstance(call.args[0], tvm.relay.Tuple):
            config[tmp]["quantized"] = False
            arg_tmp = node_id[call.args[0]]
            config[arg_tmp]["quantized"] = False

        # control strategy level
        if "level" in quantize_config:
            config[tmp]["level"] = quantize_config["level"]

    def adjust_arg_quantize_dtype(self, node):
        """adjust quantize dtype according to args"""

        need_sync_dtype = False
        consistent_dtype = None

        node_name = helper_get_call_name(node)

        for index, item in enumerate(node.args):
            if self.node_id[item] not in self.config:
                continue
            arg_config = self.config[self.node_id[item]]
            if "quantized" in arg_config and not arg_config["quantized"]:
                need_sync_dtype = True
                consistent_dtype = "int8"
                break
            if isinstance(item, tvm.relay.Call):
                name = helper_get_call_name(item)

                if self.expr_ref_count[item] > 1:
                    need_sync_dtype = True
                    consistent_dtype = "int8"
                    break

                if name in VU_OP_LIST or node_name in VU_OP_LIST:
                    need_sync_dtype = True
                    consistent_dtype = "int8"
                    break

                if name in INT8_OP_LIST or node_name in VU_OP_LIST:
                    need_sync_dtype = True
                    consistent_dtype = "int8"
                    break

                if name == "nn.relu":
                    arg_arg_name = helper_get_call_name(item.args[0])
                    if arg_arg_name in INT8_OP_LIST:
                        need_sync_dtype = True
                        consistent_dtype = "int8"
                        break

                if node_name in ELT_WISE_OP_LIST:
                    config = self.config[self.node_id[node]]["default_config"]
                    input_config = config["input{}".format(index)]
                    if input_config["dtype"] == "int4":
                        need_sync_dtype = True
                        consistent_dtype = "int4"
                elif node_name in DTYPE_AS_SAME_AS_INPUT_OP_LIST:
                    arg_node = node.args[0]

                    if arg_node in self.node_id and self.node_id[node] in self.config:
                        arg_config = self.config[self.node_id[arg_node]][
                            "default_config"
                        ]
                        arg_input0_config = arg_config["input0"]
                        need_sync_dtype = True
                        consistent_dtype = arg_input0_config["dtype"]
                    break

                elif (
                    isinstance(item, tvm.relay.Constant)
                    and node_name in ELT_WISE_OP_LIST
                ):
                    need_sync_dtype = True
                    consistent_dtype = "int8"
                    break

        if need_sync_dtype:
            assert consistent_dtype is not None
            config = self.config[self.node_id[node]]["default_config"]
            for index in range(len(node.args)):
                input_config = config["input{}".format(index)]
                input_config["dtype"] = consistent_dtype

    def adjust_arg_dtype_for_tuple(self, tup):
        """adjust quantize dtype according to args for tuple"""

        need_sync_dtype = False
        consistent_dtype = None
        for index, item in enumerate(tup.fields):
            if self.node_id[item] not in self.config:
                continue
            arg_config = self.config[self.node_id[item]]
            if "quantized" in arg_config and not arg_config["quantized"]:
                need_sync_dtype = True
                consistent_dtype = "int8"
                break
            if isinstance(item, tvm.relay.Call):
                name = helper_get_call_name(item)

                if self.expr_ref_count[item] > 1:
                    need_sync_dtype = True
                    consistent_dtype = "int8"
                    break

                if name in VU_OP_LIST:
                    need_sync_dtype = True
                    consistent_dtype = "int8"
                    break

                if name in INT8_OP_LIST:
                    need_sync_dtype = True
                    consistent_dtype = "int8"
                    break

                if name == "nn.relu":
                    arg_arg_name = helper_get_call_name(item.args[0])
                    if arg_arg_name in INT8_OP_LIST:
                        need_sync_dtype = True
                        consistent_dtype = "int8"
                        break

        if need_sync_dtype:
            assert consistent_dtype is not None
            config = self.config[self.node_id[tup]]["default_config"]
            for index, item in enumerate(tup.fields):
                input_config = config["input{}".format(index)]
                input_config["dtype"] = consistent_dtype

                if isinstance(item, tvm.relay.Call):
                    arg_config = self.config[self.node_id[item]]["default_config"]
                    if arg_config["input0"]["dtype"] == "int4":
                        arg_config["input0"]["dtype"] = consistent_dtype

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)

        name = helper_get_call_name(call)

        self.idx = self.idx + 1
        LOGGER.debug("[config] call %s %d", name, self.idx)
        if name not in self.all_op:
            self.all_op.append(name)

        tmp = self.node_id[call]

        call_id_prefix = int(tmp.split("_")[0])
        self.config[tmp] = {"valid_config": {}, "default_config": {}}

        arg_idx = -1
        args = call.args[1] if name.startswith("call_tir") else call.args
        for arg in args:
            arg_node_id = self.node_id[arg]

            arg_idx = arg_idx + 1
            arg_key = "input" + str(arg_idx)

            if isinstance(arg, tvm.relay.Constant):
                tmp_arg_dict = self.get_constant_arg_config(
                    arg_idx, name, self.quantize_config
                )
            elif isinstance(
                arg, (tvm.relay.Call, tvm.relay.Var, tvm.relay.TupleGetItem)
            ):
                tmp_arg_dict = self.get_call_arg_config(
                    arg,
                    arg_node_id,
                    self.net_in_dtype,
                    self.config,
                    self.quantize_config,
                    self.net_in_scale,
                )
            else:
                tmp_arg_dict = {
                    "default_config": {
                        "threshold": Threshold.Percentile,
                        "method": Method.Symmetry,
                        "dtype": DataType.Int8,
                        "pertensor": False,
                    }
                }
            self.config[tmp]["valid_config"].update(
                {
                    arg_key: tmp_arg_dict["default_config"]
                    if "valid_config" in tmp_arg_dict
                    else {}
                }
            )
            self.config[tmp]["default_config"].update(
                {arg_key: tmp_arg_dict["default_config"]}
            )

        self.get_whether_quantized(
            self.quantize_config, name, call_id_prefix, tmp, call, self.config
        )

        self.get_op_parameter(
            self.quantize_config, self.config, tmp, call, self.node_id
        )

        if self.quantize_config.get("int4_enable", False):
            self.adjust_arg_quantize_dtype(call)

    def visit_tuple(self, tup):
        for arg in tup.fields:
            self.visit(arg)

        tmp = self.node_id[tup]
        name = "Tuple"
        tuple_id_prefix = int(tmp.split("_")[0])

        self.idx = self.idx + 1
        LOGGER.debug("[config] tuple %d", self.idx)
        self.config[tmp] = {"valid_config": {}, "default_config": {}}

        arg_idx = -1
        for arg in tup.fields:
            arg_node_id = self.node_id[arg]

            arg_idx = arg_idx + 1
            arg_key = "input" + str(arg_idx)

            if isinstance(arg, tvm.relay.Constant):
                tmp_arg_dict = self.get_constant_arg_config(
                    arg_idx, name, self.quantize_config
                )
            elif isinstance(
                arg, (tvm.relay.Call, tvm.relay.Var, tvm.relay.TupleGetItem)
            ):
                tmp_arg_dict = self.get_call_arg_config(
                    arg,
                    arg_node_id,
                    self.net_in_dtype,
                    self.config,
                    self.quantize_config,
                    self.net_in_scale,
                )
            else:
                tmp_arg_dict = {
                    "default_config": {
                        "threshold": Threshold.Percentile,
                        "method": Method.Symmetry,
                        "dtype": DataType.Int8,
                        "pertensor": False,
                    }
                }
            self.config[tmp]["valid_config"].update(
                {
                    arg_key: tmp_arg_dict["default_config"]
                    if "valid_config" in tmp_arg_dict
                    else {}
                }
            )
            self.config[tmp]["default_config"].update(
                {arg_key: tmp_arg_dict["default_config"]}
            )

        self.get_whether_quantized(
            self.quantize_config, name, tuple_id_prefix, tmp, tup, self.config
        )

        self.get_op_parameter(self.quantize_config, self.config, tmp, tup, self.node_id)

        if self.quantize_config.get("int4_enable", False):
            self.adjust_arg_dtype_for_tuple(tup)

    def visit_function(self, fn):
        if "analysis" in tvm.relay.__dict__:
            from .tools import GetExprRefCount

            self.expr_ref_count = GetExprRefCount(fn).ret_ref_cnt
        else:
            self.expr_ref_count = tvm.relay.ir_pass.get_expr_ref_count(fn)
        return super().visit_function(fn)


def config_space(cls):
    """config model"""
    cls.final_conv_list = GetFinalConvIdx(
        cls.pre_processed_mod, cls.node_id
    ).find_call_idx
    log_value = "[config] Idx of final conv/dense in opt-model is: " + str(
        cls.final_conv_list
    )
    LOGGER.info(log_value)

    if (
        "float_list" in cls.quantize_config
        and len(cls.quantize_config["float_list"]) > 0
        and "final" in cls.quantize_config["float_list"]
    ):
        for call_idx in cls.final_conv_list:
            cls.quantize_config["float_list"].append(call_idx)
        cls.quantize_config["float_list"].remove("final")

    if (
        "float_list" in cls.quantize_config
        and len(cls.quantize_config["float_list"]) > 0
        and "all" in cls.quantize_config["float_list"]
    ):
        for call_idx in cls.quan_op:
            cls.quantize_config["float_list"].append(call_idx)
        cls.quantize_config["float_list"].remove("all")

    # support range_x1_y1_x2_y2
    range_float_op = None
    if (
        "float_list" in cls.quantize_config
        and len(cls.quantize_config["float_list"]) > 0
        and len(cls.quan_op) > 0
    ):
        for mem_op in cls.quantize_config["float_list"]:
            if isinstance(mem_op, str) and mem_op.startswith("range"):
                cls.quantize_config["float_list"].remove(mem_op)
                range_float_op = mem_op
                break
    if range_float_op is not None:
        float_range_list = range_float_op.split("_")[1:]
        assert len(float_range_list) % 2 == 0, "float_list range must in pairs"
        part_num = int(len(float_range_list) / 2)
        for part in range(part_num):
            start = int(float_range_list[2 * part])
            end = int(float_range_list[2 * part + 1])
            for quan_op_mem in cls.quan_op:
                if quan_op_mem >= start and quan_op_mem <= end:
                    cls.quantize_config["float_list"].append(quan_op_mem)

    if "float_list" in cls.quantize_config:
        log_value = (
            "[config] set op " + str(cls.quantize_config["float_list"]) + " use fp16!!!"
        )
        LOGGER.info(log_value)

    tmp = ConfigSpace(
        cls.pre_processed_mod,
        cls.node_id,
        cls.quantize_config,
        cls.net_in_dtype,
        cls.net_in_scale,
    )
    cls.config_space = tmp.config
    cls.all_op = tmp.all_op


class WinogradRevert(ExprMutator):
    """WinogradRevert"""

    def __init__(self, mod, node_id, space_cfg):
        super().__init__()
        self.node_id = node_id
        self.config_space = space_cfg
        self.winograd_revert_list = []

        if isinstance(mod, tvm.relay.Function):
            temp = self.visit(mod)
            if len(self.winograd_revert_list) > 0:
                self.new_mod = tvm.relay.ir_pass.infer_type(temp)
            else:
                self.new_mod = mod
        else:
            temp = self.visit(mod["main"])
            if len(self.winograd_revert_list) > 0:
                self.new_mod = tvm.relay.transform.InferType()(temp)
            else:
                self.new_mod = mod

    def visit_call(self, call):
        visited = super().visit_call(call)

        name = helper_get_call_name(call)
        if (
            name == "nn.contrib_conv2d_winograd1d_without_weight_transform"
            and call in self.node_id
        ):
            this_id = self.node_id[call]
            this_config = self.config_space[this_id]
            this_dtype = this_config["default_config"]["input0"]["dtype"]
            cond1 = (
                this_dtype.startswith("int")
                and int(this_dtype[3:]) >= 9
                and int(this_dtype[3:]) <= 16
            )
            cond2 = "quantized" in this_config and this_config["quantized"] is False

            if not cond1 and not cond2:
                return visited

            self.winograd_revert_list.append(int(this_id.split("_")[0]))

            attrs = {}
            attrs["strides"] = call.attrs.strides
            attrs["padding"] = call.attrs.padding
            attrs["dilation"] = call.attrs.dilation
            attrs["groups"] = call.attrs.groups
            attrs["channels"] = call.attrs.channels
            attrs["kernel_size"] = call.attrs.kernel_size
            attrs["data_layout"] = call.attrs.data_layout
            attrs["kernel_layout"] = call.attrs.kernel_layout
            attrs["out_dtype"] = call.attrs.out_dtype
            attrs["out_layout"] = call.attrs.out_layout

            arg0 = visited.args[0]
            arg1_np = visited.args[1].data.asnumpy()
            g_data = np.array(
                [
                    [1, 0, 0],
                    [1.0 / 2, 1.0 / 2, 1.0 / 2],
                    [1.0 / 2, -1.0 / 2, 1.0 / 2],
                    [0, 0, 1],
                ],
                dtype=arg1_np.dtype,
            )
            u_data, s_data, v_data = np.linalg.svd(g_data, full_matrices=False)
            inv_data = np.matmul(v_data.T * 1 / s_data, u_data.T)
            arg1_new = np.matmul(arg1_np, inv_data.T)
            arg1 = tvm.relay.Constant(tvm.nd.array(arg1_new))

            conv2d = tvm.relay.nn.conv2d(arg0, arg1, **dict(attrs))

            if "ir_pass" in tvm.relay.__dict__:
                from tvm.relay import _base

                if call.span:
                    _base.set_span(conv2d, call.span)
            else:
                from .tools import set_span

                if call.span:
                    conv2d = set_span(conv2d, call.span)

            return conv2d

        if name == "conv2d_winograd1d_bias_add" and call in self.node_id:
            this_id = self.node_id[call]
            this_config = self.config_space[this_id]
            this_dtype = this_config["default_config"]["input0"]["dtype"]
            cond1 = (
                this_dtype.startswith("int")
                and int(this_dtype[3:]) >= 9
                and int(this_dtype[3:]) <= 16
            )
            cond2 = "quantized" in this_config and this_config["quantized"] is False

            if not cond1 and not cond2:
                return visited

            self.winograd_revert_list.append(int(this_id.split("_")[0]))
            a_0 = tvm.relay.var("arg0_")
            a_1 = tvm.relay.var("arg1_")
            a_2 = tvm.relay.var("arg2_")

            temp = []

            def fvisit(expr):
                if isinstance(expr, tvm.relay.Call) and expr != visited:
                    temp.append(expr)

            if "analysis" in tvm.relay.__dict__:
                tvm.relay.analysis.post_order_visit(visited.op, fvisit)
            else:
                tvm.relay.ir_pass.post_order_visit(visited.op, fvisit)

            attrs = {}
            attrs["strides"] = temp[0].attrs.strides
            attrs["padding"] = temp[0].attrs.padding
            attrs["dilation"] = temp[0].attrs.dilation
            attrs["groups"] = temp[0].attrs.groups
            attrs["channels"] = temp[0].attrs.channels
            attrs["kernel_size"] = temp[0].attrs.kernel_size
            attrs["data_layout"] = temp[0].attrs.data_layout
            attrs["kernel_layout"] = temp[0].attrs.kernel_layout
            attrs["out_dtype"] = temp[0].attrs.out_dtype
            attrs["out_layout"] = temp[0].attrs.out_layout

            conv2d = tvm.relay.nn.conv2d(a_0, a_1, **dict(attrs))
            if "ir_pass" in tvm.relay.__dict__:
                from tvm.relay import _base
                from tvm._ffi.node import convert_to_node

                if temp[0].span is not None:
                    _base.set_span(conv2d, temp[0].span)

                bias_add = tvm.relay.nn.bias_add(conv2d, a_2, **dict(temp[1].attrs))
                new_fn = tvm.relay.Function(
                    [a_0, a_1, a_2],
                    bias_add,
                    None,
                    None,
                    tvm._api_internal._Attr(
                        "Composite",
                        convert_to_node("conv2d_bias_add"),
                        "Primitive",
                        convert_to_node(1),
                    ),
                )

            else:
                from .tools import set_span

                if temp[0].span:
                    conv2d = set_span(conv2d, temp[0].span)
                bias_add = tvm.relay.nn.bias_add(conv2d, a_2, **dict(temp[1].attrs))
                new_fn = tvm.relay.Function([a_0, a_1, a_2], bias_add)
                new_fn = new_fn.with_attr("Composite", "conv2d_bias_add")
                new_fn = new_fn.with_attr("Primitive", 1)

            arg0 = visited.args[0]
            arg1_np = visited.args[1].data.asnumpy()
            g_data = np.array(
                [
                    [1, 0, 0],
                    [1.0 / 2, 1.0 / 2, 1.0 / 2],
                    [1.0 / 2, -1.0 / 2, 1.0 / 2],
                    [0, 0, 1],
                ],
                dtype=arg1_np.dtype,
            )
            u_data, s_data, v_data = np.linalg.svd(g_data, full_matrices=False)
            inv_data = np.matmul(v_data.T * 1 / s_data, u_data.T)
            arg1_new = np.matmul(arg1_np, inv_data.T)
            arg1 = tvm.relay.Constant(tvm.nd.array(arg1_new))

            arg2 = visited.args[2]
            new_call = tvm.relay.Call(new_fn, [arg0, arg1, arg2])
            if call.span is not None:
                _base.set_span(new_call, new_call.span)
            return new_call

        return visited


def winograd_revert(cls):
    tmp = WinogradRevert(cls.pre_processed_mod, cls.node_id, cls.config_space)
    cls.pre_processed_mod = tmp.new_mod
    cls.winograd_revert = tmp.winograd_revert_list
