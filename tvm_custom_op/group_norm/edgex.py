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
"""TVM EdgeX"""
# pylint: disable-msg=C0103,W0401,W0614,C0330,R0913,R0914,C0415,R1702,E1129
import json
import os
import base64
import time
import logging
import psutil
import onnx
import numpy as np
import tvm
from tvm import relay
from tvm._ffi.runtime_ctypes import Device
from tvm.contrib.edgex.arith.nlfc import extract_nlfc_params
from tvm.contrib.edgex.tir.transform.low_level import (
    LowerCodeSplitting,
    LowerUbWoInst,
    LowerNNPDMAPass,
    FlattenMemoryPass,
    StorageRewrite,
    MisceBeforeToRuntime,
    LowerBufferDecl,
    SimplifyIR,
    LowerVIF,
    LowerLowLevelIns,
    LowerScalarOp,
)
from tvm.contrib.edgex.tir.transform.low_level.transform import (
    LowerNlfcTableAndPostRewrite,
    LowerNlfcTableIns,
)
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.transform import PlanDevices
from .config import EdgexConfig
from . import _ffi_api as _edgex_ffi_api

logger = logging.getLogger(name="edgex")


@tvm.instrument.pass_instrument
class EdgeXPassInstrument:
    """Pass instrument for debug purpose."""

    def run_before_pass(self, mod, info):
        logger.debug(f"Running before pass: {info.name}\n{mod.script()}")

    def run_after_pass(self, mod, info):
        logger.debug(f"Running after pass: {info.name}\n{mod.script()}")


def get_edgex_debug_working_dir():
    """Get debug working directory"""
    working_dir = _edgex_ffi_api.GetEdgeXDebugWorkingDir()
    if working_dir:
        working_dir = str(working_dir)
    return working_dir


def is_llvm_dump_enabled():
    """Whether enable llvm debug dumping."""
    return _edgex_ffi_api.IsLLVMDumpEnabled()


def get_input_batch(mod):
    """Extract batch num from the first conv in the model."""

    class BatchGetter(relay.ExprVisitor):
        """Extract batch num from the first conv in the model."""

        def __init__(self):
            self._batch = 1
            self._found_conv = False
            super().__init__()

        def __call__(self, func):
            self.visit(func)
            return self._batch

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op) and "conv" in call.op.name:
                self._batch = call.checked_type.shape[0]
                self._found_conv = True
            return super().visit_call(call)

    getter = BatchGetter()
    input_batch = getter(relay.transform.InferType()(mod)["main"])
    if not getter._found_conv:
        input_batch = mod["main"].params[0].type_annotation.shape[0]
    return input_batch


def build_config_nnp(
    config=None, extra_disabled_pass=None, opt_level=2, instruments=None, for_dumper=False
):
    """Add nnp lower pass.

    Returns
    -------
    ret : tvm.transform.PassContext
        The pass context contains the nnp lower pass.
    """

    default_relay_config = {
        "relay.fallback_device_type": Device.kDLEdgeX,
        "relay.backend.use_multitarget_pass_context": True,
        "relay.edgex.global_channel_padding": False,
        "edgex.FuseSubgraph.max_fused_nu_num": 16,
        "edgex.FuseSubgraph.max_fused_vu_num": 16,
        "edgex.FuseSubgraph.dm_limit_multiple": 8,
        "edgex.FuseSubgraph.skip_fused": False,
        "relay.edgex.use_fused_span": True,
        "edgex.batch_num": 1,
        "relay.edgex.byoa": False,
        "relay.attention.rewrite": True,
    }

    default_tir_config = {
        "tir.edgex.EstimateCost.enable": False,
        "tir.edgex.CalculateMac.enable": True,
        "tir.edgex.EstimateVcuCycles.enable": True,
        "tir.edgex.EstimateVcuCycles.verbose": False,
        "tir.LoopPartition": {
            "unroll_loop_with_partition_hint_no_interval": True,
        },
        "tir.edgex.InjectHandshake.config": {
            "dump_debug_info": False,
        },
        "tir.edgex.StorageRewriteNNP400.config": {
            "dump_memory_footprint": False,
            "use_experimental_soma_dm_allocator": True,
        },
        "tir.edgex.StorageRewriteNNP400.verbose": False,
        "tir.edgex.InjectDmaIntrin.verbose": False,
        "tir.edgex.InjectDebugIntrinsic.enable": False,
        "tir.edgex.DetectEdgeXAnomaly.enable": True,
        "tir.edgex.MergeParams.enable": True,
        "tir.edgex.InplaceReshape.enable": True,
        "tir.edgex.EliminateCondition.enable": True,
        "tir.edgex.InjectCheckpoint.enable": True,
        "tir.edgex.InjectCheckpoint.enable_vxdma_tracing": False,
        "tir.edgex.InjectIfSync.enable": True,
        "tir.edgex.InjectIcachePrefetch.enable": opt_level >= 2,
    }

    default_parallelism_config = {
        "edgex.compile_thread.schedule": 0,  # 目前调度多线程有问题
        "edgex.compile_thread.lower": psutil.cpu_count(logical=False),
        "edgex.compile_thread.tir_to_runtime": psutil.cpu_count(logical=False),
    }

    default_config = {}
    default_config.update(default_relay_config)
    default_config.update(default_tir_config)
    default_config.update(default_parallelism_config)
    if config is not None:
        default_config.update(config)

    from .lower_config import create_default_nnp_pass_list

    default_config["tir.create_pass_list"] = lambda: create_default_nnp_pass_list(
        for_dumper=for_dumper, config=default_config
    )

    disabled_pass = [
        # we will use tvm_access_ptr() in codegen and do not lower it to address_of()
        "tir.LowerDeviceStorageAccessInfo",
        "tir.StorageRewrite",
        "SimplifyInference",
        "SimplifyExpr",
        "SimplifyExprPostAlterOp",
    ]
    if extra_disabled_pass is not None:
        if not isinstance(extra_disabled_pass, list):
            extra_disabled_pass = [extra_disabled_pass]
        for pass_obj in extra_disabled_pass:
            if isinstance(pass_obj, tvm.transform.Pass):
                pass_name = pass_obj.info.name
            elif isinstance(pass_obj, str):
                pass_name = pass_obj
            else:
                raise ValueError("pass in `extra_disabled_pass` should be string or Pass")
            disabled_pass.append(pass_name)

    return tvm.transform.PassContext(
        config=default_config,
        disabled_pass=disabled_pass,
        opt_level=opt_level,
        instruments=instruments,
    )


def build_edgex_default_config(mod, opt_level):
    """build edgex default pass context.

    Returns
    -------
    config : dict
        The edgex pass config.
    opt_level: int
        updated opt_level especially for yolo O3
    """
    updated_opt_level = opt_level
    config = {}
    batch = get_input_batch(mod)
    config["edgex.batch_num"] = batch
    if opt_level >= 2:
        config["relay.edgex.global_channel_padding"] = True
    else:
        config["edgex.FuseSubgraph.layer_limit"] = 1

    return config, updated_opt_level


def build_ll_nnp_config(config=None, extra_disabled_pass=None, opt_level=2, instruments=None):
    """Add nnp lower pass specific to low level lang code.

    Returns
    -------
    ret : tvm.transform.PassContext
        The pass context contains the nnp lower pass.
    """

    from tvm.contrib.edgex.tir.transform import (
        AnnotatePureComputeScope,
        DumpBufferTag,
        DumpOrReuseLoweredTIR,
    )

    pass_list = []

    pass_list.append((0, LowerNlfcTableAndPostRewrite()))
    pass_list.append((0, LowerUbWoInst()))
    pass_list.append((1, LowerNNPDMAPass()))
    pass_list.append((2, LowerVIF()))
    pass_list.append((2, LowerLowLevelIns()))

    dump_tir = os.environ.get("EDGEX_DEBUG_DUMP_TIR", None) is not None
    working_dir = os.environ.get("EDGEX_DEBUG_WORKING_DIR", None)
    if dump_tir:
        if working_dir is None:
            raise ValueError("Please specify EDGEX_DEBUG_WORKING_DIR for tir debug purpose")
        pass_list.append((2, DumpBufferTag()))

    pass_list.append((2, LowerCodeSplitting()))
    pass_list.append((2, FlattenMemoryPass()))
    pass_list.append((2, StorageRewrite()))
    pass_list.append((2, MisceBeforeToRuntime()))
    pass_list.append((2, AnnotatePureComputeScope()))
    pass_list.append((3, LowerBufferDecl()))
    pass_list.append((3, LowerScalarOp()))
    pass_list.append((3, SimplifyIR()))

    # lowered tir dumping support
    use_existing_tir = os.environ.get("EDGEX_DEBUG_USE_EXISTING_TIR", None) is not None
    if dump_tir or use_existing_tir:
        if working_dir is None:
            raise ValueError("Please specify EDGEX_DEBUG_WORKING_DIR for tir debug purpose")
        pass_list.append((3, DumpOrReuseLoweredTIR(working_dir, use_existing_tir)))

    default_config = {
        "tir.add_lower_pass": pass_list,
        "tir.disable_cse_tir": True,
        "relay.fallback_device_type": Device.kDLEdgeX,
        "relay.backend.use_multitarget_pass_context": True,
        "tir.edgex.EstimateCost.enable": False,
        "tir.edgex.CalculateMac.enable": True,
        "tir.LoopPartition": {
            "unroll_loop_with_partition_hint_no_interval": True,
        },
        "tir.edgex.InjectDmaIntrin.verbose": False,
        "tir.edgex.InjectDebugIntrinsic.enable": False,
        "tir.edgex.DetectEdgeXAnomaly.enable": True,
        "tir.edgex.MergeParams.enable": False,
        "relay.edgex.global_channel_padding": False,
        "edgex.FuseSubgraph.max_fused_nu_num": 16,
        "edgex.FuseSubgraph.max_fused_vu_num": 16,
        "edgex.FuseSubgraph.dm_limit_multiple": 8,
        "edgex.FuseSubgraph.skip_fused": False,
        "tir.edgex.InjectCheckpoint.enable": True,
        "tir.edgex.EliminateCondition.enable": True,
    }
    if config is not None:
        default_config.update(config)

    disabled_pass = [
        # we will use tvm_access_ptr() in codegen and do not lower it to address_of()
        "tir.LowerDeviceStorageAccessInfo",
        "tir.StorageRewrite",
        "tir.InjectPrefetch",
        "tir.TextureFlatten",
        "tir.StorageFlatten",
        "tir.LowerCrossThreadReduction",
        "tir.LowerInitBlock",
        "tir.PlanAndUpdateBufferAllocationLocation",
        "tir.ConvertBlocksToOpaque",
        "tir.UnifyThreadBinding",
        "tir.ManifestSharedMemoryLocalStage",
        "tir.CompactBufferAllocation",
        "tir.LowerMatchBuffer",
        "tir.InjectSoftwarePipeline",
        "tir.LowerOpaqueBlock",
        "tir.LoopPartition",
        "tir.TrimTensor",
        "tir.FlattenBuffer",
        "tir.BF16Legalize",
        "tir.NarrowDataType",
        "tir.VectorizeLoop",
        "tir.InjectVirtualThread",
        "tir.InjectDoubleBuffer",
        "tir.LowerAsyncDMA",
        "tir.UnrollLoop",
        "tir.RenormalizeSplitPattern",
        "tir.Simplify",
        "tir.RemoveNoOp",
        "tir.RewriteUnsafeSelect",
        "tir.HoistIfThenElse",
    ]
    if extra_disabled_pass is not None:
        if not isinstance(extra_disabled_pass, list):
            extra_disabled_pass = [extra_disabled_pass]
        for pass_obj in extra_disabled_pass:
            if isinstance(pass_obj, tvm.transform.Pass):
                pass_name = pass_obj.info.name
            elif isinstance(pass_obj, str):
                pass_name = pass_obj
            else:
                raise ValueError("pass in `extra_disabled_pass` should be string or Pass")
            disabled_pass.append(pass_name)
    return tvm.transform.PassContext(
        config=default_config,
        disabled_pass=disabled_pass,
        opt_level=opt_level,
        instruments=instruments,
    )


@tvm.register_func("tvm.edgex.low_level.lower_nlfc_table_ins_and_post_rewrite")
def nlfc_table_ins_lower(original_func):
    """lower get_nlfc_table ins in Low Level Programming."""

    transformed_func = LowerNlfcTableIns()(IRModule({"main": original_func}))["main"]

    nlfc_params, nlfc_arrs, transformed_func = extract_nlfc_params(transformed_func)
    preproc_argument_rewrite_mgr = None
    nlfc_buffers = []
    if nlfc_params:
        for p in nlfc_params:
            nlfc_buffers.append(transformed_func.buffer_map[p])

        # add attr "NlfcTableParams" back for cascade schedule
        transformed_func = transformed_func.with_attr("NlfcTableParams", nlfc_params)

        # trace pre-schedule operations relay argument updates
        forward = lambda: [relay.Constant(_) for _ in nlfc_arrs]
        backward = lambda *x: []
        from tvm.contrib.edgex.relay.transform.argument_rewrite_manager import (
            ArgumentRewriteManager,
        )

        preproc_argument_rewrite_mgr = ArgumentRewriteManager(original_func)
        preproc_argument_rewrite_mgr.trace_update(
            [], nlfc_buffers, transformed_func, forward, backward
        )
    if preproc_argument_rewrite_mgr:
        rewrite_mod = preproc_argument_rewrite_mgr.realize()
        json_str = tvm.ir.save_json(rewrite_mod)
        json_str = json.dumps(json.loads(json_str), separators=(",", ":"))
        func = transformed_func.with_attr("post_schedule_argument_rewrite", json_str)

        return func
    else:
        return original_func


@tvm.target.override_native_generic_func("tvm.edgex.get_tir_pass_context")
def get_tir_pass_context():
    """Generic func to get PassContext to use with tir lowering on current target."""
    return PassContext.current()


@get_tir_pass_context.register(["edgex"])
def get_edgex_tir_pass_context():
    """EdgeX specific PassContext to use with tir lowering."""
    if len(PassContext.current().config) > 0:
        return PassContext.current()
    else:
        return build_config_nnp()


@get_tir_pass_context.register(["cpu", "arm_cpu", "edgex_virtual_host"])
def get_cpu_tir_pass_context():
    """EdgeX specific PassContext to use with tir lowering, for host cpu op."""
    return PassContext()


@tvm.register_func("tvm.edgex.low_level.get_tir_pass_config")
def get_low_level_tir_pass_config():
    """Get the building pass config of low level programming lang."""
    return build_ll_nnp_config()


@tvm.register_func("tvm.info.mem.dm")
def mem_info_edgex_dm_buffer():
    """Mem info registry for dm"""
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=8,
        max_simd_bits=4096 * 8,
        max_num_bits=int(EdgexConfig.get_current().DM_SIZE) * 8,
        head_address=None,
    )


@tvm.register_func("tvm.info.mem.vm")
def mem_info_edgex_vm_buffer():
    """Mem info registry for vm"""
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=8,
        max_simd_bits=4096 * 8,
        max_num_bits=int(EdgexConfig.get_current().VM_SIZE) * 8,
        head_address=None,
    )


####################
# Doc api for edgex
####################
def load_model_from_file(model_path, model_format, shape_dict, **kwargs):
    """
    Load model file from str path, currently only onnx models are supported.

    Parameters
    ==========
    model_path: str
        Path to model file.
    model_format: str
        The framework used by model, currently only onnx models are supported.
    shape_dict: dict of str to int list/tuple
        A dict of string/int tuples or lists representing shape of model input.
    kwargs: dict of str to dict of str to str
        A dict of string/dict representing data type of model input.

    Returns
    =======
    mod: tvm.ir.module.IRModule
        tvm IRModule
    params: dict of str to tvm.NDArray.array
        The params of IRModule.
    """

    if not isinstance(model_path, (str, list, tuple)):
        raise RuntimeError("model_path should be a string or a list/tuple.")
    if not isinstance(model_format, str):
        raise RuntimeError("model_format should be a string.")
    if not isinstance(shape_dict, dict):
        raise RuntimeError("shape_dict should be a dict.")
    if not isinstance(kwargs, dict):
        raise RuntimeError("kwargs should be a dict.")

    if "onnx" in model_format:
        if "qnn" in kwargs and kwargs["qnn"]:
            optimize_qnn_model = getattr(tvm.contrib.edgex.relay.qnn, "optimize_qnn_model")
            mod, params = optimize_qnn_model(model_path, model_format, shape_dict, **kwargs)
            return mod, params
        else:
            load_model = getattr(tvm.contrib.edgex.relay.frontend.onnx, "load_model")
            mod, params = load_model(model_path, shape_dict, **kwargs)
    elif model_format == "mxnet":
        import mxnet as mx

        mx_sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch=0)
        mod, params = relay.frontend.from_mxnet(
            mx_sym, shape_dict, arg_params=arg_params, aux_params=aux_params, **kwargs
        )
    elif model_format == "caffe":
        from google.protobuf import text_format
        from tvm.contrib.edgex.relay.frontend import caffe_pb2 as pb

        def find_files_with_suffix(all_files, suffix):
            """find_files_with_suffix"""
            return [file for file in all_files if file.endswith(suffix)]

        init_net = pb.NetParameter()
        predict_net = pb.NetParameter()
        if isinstance(model_path, (list, tuple)) and len(model_path) == 2:
            proto_file = find_files_with_suffix(model_path, ".prototxt")[0]
            blob_file = find_files_with_suffix(model_path, ".caffemodel")[0]
        else:
            assert isinstance(model_path, str), "model_path should be a str!"
            all_files = os.listdir(model_path)
            proto_file = find_files_with_suffix(all_files, ".prototxt")
            blob_file = find_files_with_suffix(all_files, ".caffemodel")
            assert (
                len(proto_file) == 1 and len(blob_file) == 1
            ), f"NOT only one! proto_file: {proto_file}; blob_file: {blob_file}"
            proto_file = os.path.join(model_path, proto_file[0])
            blob_file = os.path.join(model_path, blob_file[0])

        with open(proto_file, "r") as f:
            text_format.Merge(f.read(), predict_net)
        with open(blob_file, "rb") as f:
            init_net.ParseFromString(f.read())

        mod, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, **kwargs)
    else:  # tensorflow/pytorch/tflite...
        if model_format == "tensorflow":
            model_format = "pb"
        load_model = getattr(tvm.driver.tvmc.frontends, "load_model")
        tvmc_model = load_model(model_path, model_format, shape_dict, **kwargs)
        mod, params = tvmc_model.mod, tvmc_model.params

    # graph-level preprocess before quantization
    mod = tvm.contrib.edgex.relay.transform.ConvertConvolutionStrides()(mod)
    mod = tvm.contrib.edgex.relay.transform.ConvertImageResize2dToDeConv(True)(mod)
    mod = tvm.contrib.edgex.relay.transform.ConvertFocusLayerToConv()(mod)
    mod = tvm.contrib.edgex.relay.transform.ConvertAttentionAddInfMask()(mod)
    mod = tvm.contrib.edgex.relay.transform.ConvertConv1dToConv2d()(mod)

    if isinstance(model_path, str) and model_path.find("e1") >= 0:  # TODO: FIXME ASAP
        mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
        from tvm.relay.quantization.relay_transforms import FuseAdd
        from tvm.relay.quantization.pre_processes.fold_scale import FoldScale
        from tvm.relay.transform import CombineParallelConv2D

        mod = FoldScale()(mod)  # pylint: disable-msg=not-callable
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.InferType()(mod)
        mod = FuseAdd()(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.InferType()(mod)
        mod = CombineParallelConv2D(min_num_branches=2)(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.InferType()(mod)
        mod, params = tvm.contrib.edgex.relay.transform.ExtractConstants()(mod)
        mod = tvm.contrib.edgex.relay.transform.SimplifySlice()(mod)

    return mod, params


def normalize_params(params):
    """
    Limit parameter range.

    Parameters
    ==========
    params: dict of str to tvm.NDArray.array or numpy.ndarray
    The params of IRModule.

    Returns
    =======
    params: dict of str to tvm.NDArray.array
        The params of IRModule after limitation.
    """

    params = dict(params.items())
    warning_flag = False
    for k in params:
        if k.find("round_right_shift") >= 0:
            norm = params[k] if isinstance(params[k], np.ndarray) else params[k].asnumpy()
            if (norm < 1).any() or (norm > 24).any():
                warning_flag = True
        elif k.find("multiply") >= 0:
            norm = params[k] if isinstance(params[k], np.ndarray) else params[k].asnumpy()
            if (norm > 127).any():
                warning_flag = True
        elif k.find("bias_add") >= 0:
            norm = params[k] if isinstance(params[k], np.ndarray) else params[k].asnumpy()
            if (norm < -(2**20)).any() or (norm > 2**20).any():
                warning_flag = True

    if warning_flag:
        logger.warning("model params out of normalize range")
    return params


def search_winograd(mod, params=None):
    """
    Transform IRModule into buildable form for nnp before final build.

    Parameters
    ==========
    mod: tvm.ir.module.IRModule
        The object of IRModule.
    params: dict of str to tvm.NDArray.array
        The params of IRModule.
    Returns
    =======
    wingrad_spans: list[str]
        List of all span names of convolutions that need to use winograd.
    """
    from tvm.contrib.edgex.testing import UnsupportedDetector, get_edgex_plan_device_config
    from tvm.contrib.edgex.relay.transform import (
        ConvertGelu,
        ConvertSplitToStridedSlice,
        ConvertDepthwiseConv2D,
        PadConvChannels,
        ConvertNULayout,
        AttentionLayoutTransform,
        LiftLayoutTransform,
        OptimizeLayoutTransform,
        FusionStitch,
        AnnotateEdgex,
        MergeFusedFunctions,
        cost_model_pattern_table,
    )
    from tvm.contrib.edgex.relay.transform.fuse_subgraph import SubgraphFusor

    EdgexConfig.set_current(
        EdgexConfig.get_current().with_cfg("relay_cost_model.winograd_search_en", True)
    )
    if len(PassContext.current().config) > 0:
        configs = dict(PassContext.current().config)
        opt_level = PassContext.current().opt_level
        configs.update({"edgex.FuseSubgraph.dump_perf_result": False})
        ctx = build_config_nnp(config=configs, opt_level=opt_level)
    else:
        raise ValueError(
            "EDGEX PassContext is not initialized, \
                use build_config_nnp to initialize the PassContext."
        )

    if params is not None:
        func_with_params = bind_params_by_name(mod["main"], params)
        mod["main"] = func_with_params

    with ctx:
        mod = ConvertGelu()(mod)
        mod = ConvertSplitToStridedSlice()(mod)
        mod = ConvertDepthwiseConv2D()(mod)
        mod = PadConvChannels()(mod)
        mod = ConvertNULayout()(mod)
        if ctx.config["relay.attention.rewrite"]:
            mod = AttentionLayoutTransform()(mod)
        mod = LiftLayoutTransform()(mod)
        mod = OptimizeLayoutTransform()(mod)
        mod = FusionStitch()(mod)
        mod = AnnotateEdgex("edgex", UnsupportedDetector())(mod)
        mod = MergeFusedFunctions(cost_model_pattern_table())(mod)
        mod = PlanDevices(get_edgex_plan_device_config(ctx))(mod)
        searcher = SubgraphFusor(mutate=False)
        searcher.run(mod)

        return searcher._winograd_layer_spans


def optimize_nnp_model(
    mod,
    params,
    target_device,
    keep_params=True,
    offload_cpu_strategy=None,
):  # pylint: disable=dangerous-default-value
    """
    Transform IRModule into buildable form for nnp before final build.

    Parameters
    ==========
    mod: tvm.ir.module.IRModule
        The object of IRModule.
    params: dict of str to tvm.NDArray.array
        The params of IRModule.
    target_device: tvm.target.Target
        If specified, use the specified target for device planning.
    keep_params: bool
        If True, the params is get preserved and returned.
        If False, the params is merged into ir module and return params as None.
    offload_cpu_strategy: function
        If specified, decide when a op should offload to cpu.

    Returns
    =======
    mod: tvm.ir.module.IRModule
        tvm IRModule
    params: dict or None
        tvm params dict. Return None if input params is None or keep_params=False
    """
    from tvm.contrib.edgex.testing import get_edgex_plan_device_config
    from tvm.contrib.edgex.relay.transform import (
        FusionStitch,
        AnnotateEdgex,
        ConvertDepthwiseConv2D,
        ConvertSumPool2D,
        ConvertSplitToStridedSlice,
        ConvertActivationQuant,
        PadConvChannels,
        LiftLayoutTransform,
        NarrowDownConcatDtype,
        ConvertNULayout,
        AttentionLayoutTransform,
        OptimizeLayoutTransform,
        # ConvertQDenseToConv,
        RewriteIndicesDtype,
        EliminateRedundantPad,
        EliminateRedundantConcat,
        ConvertMeanToSum,
        ConvertIntDivOps,
        RewriteConvertScalar,
        SliceAxisReorder,
        extract_constants,
        AddFusedFuncName,
        AddUnFusedSpan,
        FallbackRewriteI64Ops,
        FuseSubgraph,
        MergeFusedFunctions,
        rule_fusion_pattern_table,
        cost_model_pattern_table,
        SeparateConvolutionPad,
        SimplifySlice,
        ConvertPetrAttentionQDense,
        ConvertGelu,
        EliminateRedundantMultiply,
        ConvertPowerToSqrt,
        ConvertFmodToImod,
        ForkCastOps,
        ConvertImageResize2dToDeConv,
        WrapAtomicFunction,
    )

    class _MarkOptimized(relay.ExprMutator):
        def visit_function(self, fn):
            new_params = [self.visit(x) for x in fn.params]
            new_body = self.visit(fn.body)
            new_func = relay.Function(
                new_params, new_body, fn.ret_type, fn.type_params, fn.attrs, fn.span
            )
            if fn != self.main_func:
                new_func = new_func.with_attr("Optimized", 1)
            return new_func

        def __call__(self, mod):
            self.main_func = mod["main"]
            func = self.visit(mod["main"])
            mod["main"] = func
            return mod

    class _CheckOptimized(relay.ExprVisitor):
        def __init__(self, mod):
            super().__init__()
            self.main_func = mod["main"]
            self._all_optimized = []
            self.visit(mod["main"])
            if not self._all_optimized:
                self.optimized = False
            else:
                self.optimized = all(self._all_optimized)

        def visit_function(self, fn):
            for x in fn.params:
                self.visit(x)
            self.visit(fn.body)
            if fn != self.main_func:
                if "Optimized" in fn.attrs:
                    self._all_optimized.append(1)
                else:
                    self._all_optimized.append(0)

    if len(PassContext.current().config) > 0:
        ctx = PassContext.current()
    else:
        raise ValueError(
            "EDGEX PassContext is not initialized, \
                use build_config_nnp to initialize the PassContext."
        )

    if params is not None:
        func_with_params = bind_params_by_name(mod["main"], params)
        mod["main"] = func_with_params

    if not _CheckOptimized(mod).optimized:
        from tvm.contrib.edgex.relay.transform.byoa import BYOAPass

        if ctx.config["relay.edgex.byoa"]:
            mod = ConvertPetrAttentionQDense()(mod)  # TODO(cww):临时解决方案，后续需要删除
            mod = BYOAPass()(mod)

        # mod = ConvertQDenseToConv()(mod)  # 为了避免干扰BYOAPass中的attention pattern，因此放在其后
        mod = EliminateRedundantMultiply()(mod)
        mod = ConvertGelu()(mod)
        mod = ConvertSplitToStridedSlice()(mod)
        mod = SliceAxisReorder()(mod)
        mod = ConvertSumPool2D()(mod)
        mod = ConvertImageResize2dToDeConv()(mod)
        mod = ConvertDepthwiseConv2D()(mod)
        mod = PadConvChannels()(mod)
        mod = SimplifySlice()(mod)
        mod = ConvertMeanToSum()(mod)
        mod = ConvertIntDivOps()(mod)
        mod = EliminateRedundantPad()(mod)
        mod = EliminateRedundantConcat()(mod)
        mod = RewriteIndicesDtype()(mod)
        # rationale: need after all transform to convolution pass
        # mod = ConvertDilatedConvolution()(mod)  # supported skip dilation 0 (2023.12.29)
        # rationale: need after convert dilated convolution
        mod = SeparateConvolutionPad()(mod)
        mod = ConvertNULayout()(mod)
        if ctx.config["relay.attention.rewrite"]:
            mod = AttentionLayoutTransform()(mod)
        mod = LiftLayoutTransform()(mod)
        mod = OptimizeLayoutTransform()(mod)
        mod = NarrowDownConcatDtype()(mod)
        mod = ConvertActivationQuant()(mod)
        mod = FallbackRewriteI64Ops()(mod)
        mod = RewriteConvertScalar()(mod)
        mod = ConvertPowerToSqrt()(mod)
        mod = ConvertFmodToImod()(mod)
        mod = ForkCastOps()(mod)
        if not ctx.config["relay.edgex.use_fused_span"]:
            mod = AddUnFusedSpan()(mod)
        mod = FusionStitch()(mod)
        if offload_cpu_strategy is None:
            from tvm.contrib.edgex.testing import UnsupportedDetector

            offload_cpu_strategy = UnsupportedDetector()
        mod = AnnotateEdgex(target_device.kind.name, offload_cpu_strategy)(mod)
        mod = PlanDevices(get_edgex_plan_device_config(ctx, target_device))(mod)
        # FuseSubgraph must be placed after PlanDevices
        if ctx.opt_level < 2:
            mod = MergeFusedFunctions(rule_fusion_pattern_table())(mod)
        elif ctx.opt_level >= 2:
            mod = MergeFusedFunctions(cost_model_pattern_table())(mod)
            mod = PlanDevices(get_edgex_plan_device_config(ctx, target_device))(mod)
            mod = FuseSubgraph()(mod)
        mod = WrapAtomicFunction()(mod)
        # AddFusedFuncName is always executed at the end
        mod = AddFusedFuncName(ctx.config["relay.edgex.use_fused_span"])(mod)

        # mark optimization finished
        mod = _MarkOptimized()(mod)

    new_params = None
    if params is not None and keep_params:
        mod, new_params = extract_constants(mod)
        if new_params is not None and len(new_params) == 0:
            new_params = None
    return mod, new_params


def export_graph_library(graph_lib, export_lib_path, target_host_cc=None, ir_mod=None):
    """Export graph library to elf model file.

    Parameters
    ==========
    graph_lib: tvm.runtime.Module
        The compiled graph module.

    export_lib_path: str
        Output model path.

    target_host_cc: str
        Target cc executable path if specified.

    ir_mod: tvm.IRModule
        Optional graph irmodule for reference.
    """
    # collect all addon file from ir mod
    all_link_files = []
    addons = []
    cc_options = []
    if ir_mod is not None:
        for _, func in ir_mod.functions.items():
            if isinstance(func, tvm.tir.PrimFunc):
                if func.attrs is not None and "nnp_c_link_files" in func.attrs:
                    link_files = func.attrs["nnp_c_link_files"]
                    if link_files:
                        all_link_files.extend(link_files)
    for node in json.loads(graph_lib["get_graph_json"]())["nodes"]:
        node_attrs = node.get("attrs", {})
        link_files = node_attrs.get("nnp_c_link_files")
        if link_files:
            all_link_files.extend(link_files)
    for filepath in set(all_link_files):
        if filepath.endswith(".so") or filepath.endswith(".dll") or filepath.endswith(".a"):
            cc_options.append(f"-L{filepath}")
        else:
            addons.append(filepath)

    # prepare cc options
    if len(addons) > 0:
        cc_options.extend(
            ["-std=c++17", "-O2", "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>"]
        )
        tvm_root_cands = [
            tvm.__path__[0],
            os.path.join(tvm.__path__[0], "..", ".."),
            os.environ.get("TVM_HOME", "."),
        ]
        cc_options.append(f"-I/usr/local/include/eigen3")  # add eigen3
        for cand in tvm_root_cands:
            cc_options.append(f"-I{cand}/include")
            cc_options.append(f"-I{cand}/3rdparty/dlpack/include")
            cc_options.append(f"-I{cand}/3rdparty/dmlc-core/include")     
    graph_lib.export_library(export_lib_path, cc=target_host_cc, addons=addons, options=cc_options)


def _get_target_info(target_host, target_host_cc, target_device, export_lib_path):
    if target_device is None:
        target_device = tvm.target.edgex()
    elif isinstance(target_device, str):
        target_device = tvm.target.Target(target_device)
    elif not isinstance(target_device, tvm.target.Target):
        raise RuntimeError("target_device should be str, tvm.target.Target or None.")

    def __arg_to_list(arg):
        if isinstance(arg, (list, tuple)):
            return arg
        return [arg]

    target_host_list = __arg_to_list(target_host)
    target_host_cc_list = __arg_to_list(target_host_cc)
    export_lib_path_list = __arg_to_list(export_lib_path)

    host_num = len(target_host_list)
    if host_num > 1:
        cpu_target = tvm.target.Target("edgex_virtual_host")
    elif target_host_list[0]:
        cpu_target = target_host_list[0]
    else:
        cpu_target = tvm.target.Target("llvm")
    if target_device.host is None:
        target_device = tvm.target.Target("edgex", host=cpu_target)

    return (
        cpu_target,
        target_device,
        host_num,
        target_host_list,
        target_host_cc_list,
        export_lib_path_list,
    )


def compile_nnp_model(
    mod,
    params,
    export_lib_path="",
    target_host=None,
    target_host_cc=None,
    target_device=None,
):
    """
    Compile nnp IRModule.

    Parameters
    ==========
    mod: tvm.ir.module.IRModule
        The object of IRModule.
    params: dict of str to tvm.NDArray.array
        The params of IRModule.
    export_lib_path: str or list of str.
        If specified, export compiled model to path.
    target_host: list of str or tvm.target.Target
        Host targets specification, None for default host.
    target_host_cc: list of str
        Host targets cc toolchain path, None for default host cc.
    target_device: str or tvm.target.Target
        Device target.

    Returns
    =======
    lib: tvm.runtime.Module or List[tvm.runtime.Module]
        Tvm graph runtime libs for each host.
    """

    from tvm.contrib.edgex.runtime.edgex_graph_executor import (
        package_graph_lib_components,
        extract_graph_lib_components,
        build_virtual_host_module,
    )

    (
        cpu_target,
        target_device,
        host_num,
        target_host_list,
        target_host_cc_list,
        export_lib_path_list,
    ) = _get_target_info(target_host, target_host_cc, target_device, export_lib_path)

    if len(PassContext.current().config) > 0:
        ctx = PassContext.current()
    else:
        raise ValueError(
            "EDGEX PassContext is not initialized, \
                use build_config_nnp to initialize the PassContext."
        )

    with ctx:
        graph_lib = tvm.relay.build(
            mod, params=params, target={"edgex": target_device, "cpu": cpu_target}
        )

    graph_json, graph_params, host_mod, dev_mod = extract_graph_lib_components(graph_lib)

    if dev_mod is None:
        # certain onchip toolkit version require device module must exist,
        # thus if there are no device ops, create a empty one.
        dev_mod = tvm.get_global_func("target.build.edgex")(tvm.IRModule(), target_device)

    libs = []
    for i in range(host_num):
        target_host = target_host_list[i]
        target_host_cc = target_host_cc_list[i] if i < len(target_host_cc_list) else None
        export_lib_path = export_lib_path_list[i] if i < len(export_lib_path_list) else None
        if not isinstance(target_host, (str, tvm.target.Target, type(None))):
            raise RuntimeError("target_host should be string or tvm.target.Target or None.")
        if not isinstance(target_host_cc, (str, type(None))):
            raise RuntimeError("target_host_cc should be string or None.")
        if not isinstance(export_lib_path, (str, type(None))):
            raise RuntimeError("export_lib_path should be a string.")
        if host_num > 1:
            actual_host_mod = build_virtual_host_module(host_mod, target_host)
            graph_lib = package_graph_lib_components(
                graph_json, graph_params, actual_host_mod, dev_mod
            )

        if export_lib_path is not None and export_lib_path != "":
            export_graph_library(
                graph_lib, export_lib_path, target_host_cc=target_host_cc, ir_mod=mod
            )
        libs.append(graph_lib)

    return libs[0] if host_num == 1 else libs


def optimize_and_compile(
    mod,
    params,
    working_dir="",
    export_lib_path="",
    opt_level=2,
    offload_cpu_strategy=None,
    target_host=None,
    target_host_cc=None,
    target_device=None,
    config=None,
):
    """
    Compile nnp IRModule.

    Parameters
    ==========
    mod: tvm.ir.module.IRModule
        The object of IRModule.
    params: dict of str to tvm.NDArray.array
        The params of IRModule.
    working_dir: str
        If specified, output compilation information to the working directory.
    export_lib_path: str or list of str.
        If specified, export compiled model to path.
    opt_level: int
        The optimization level.
    offload_cpu_strategy: function
        If specified, decide when a op should offload to cpu.
    target_host: list of str or tvm.target.Target
        Host targets specification, None for default host.
    target_host_cc: list of str
        Host targets cc toolchain path, None for default host cc.
    target_device: str or tvm.target.Target
        Device target.
    config: dict
        The config of PassContext.

    Returns
    =======
    lib: tvm.runtime.Module or List[tvm.runtime.Module]
        Tvm graph runtime libs for each host.
    """

    if not isinstance(mod, tvm.ir.module.IRModule):
        raise RuntimeError("mod should be a tvm.ir.module.IRModule.")
    if not isinstance(params, (dict, tvm.ir.container.Map, type(None))):
        raise RuntimeError("params should be a dict or None.")
    if not isinstance(working_dir, str):
        raise RuntimeError("working_dir should be a string.")
    if not isinstance(opt_level, int):
        raise RuntimeError("opt_level should be a int.")
    if offload_cpu_strategy is not None:
        if not callable(offload_cpu_strategy):
            raise RuntimeError("offload_cpu_strategy should be a callable object or None.")

    default_config, opt_level = build_edgex_default_config(mod, opt_level)
    default_config["edgex.relay_to_tir.collect_lower_errors"] = True
    if config is not None:
        default_config.update(config)
    if working_dir != "":
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        working_dir = os.path.join(working_dir, now)
    default_config["edgex.relay_to_tir.user_dir"] = working_dir

    # distinguish pass context config and others
    if "hardware.pe_num" in default_config:
        cfg = EdgexConfig.get_current()
        cfg = cfg.with_cfg("PE_NUM", default_config.pop("hardware.pe_num"))
        EdgexConfig.set_current(cfg)

    logger.info(f"Compile config: {default_config}")
    logger.info(f"Opt level: {opt_level}")
    pass_ctx = build_config_nnp(config=default_config, opt_level=opt_level)

    with pass_ctx:
        _, device, _, _, _, _ = _get_target_info(
            target_host, target_host_cc, target_device, export_lib_path
        )

        mod, _ = optimize_nnp_model(
            mod,
            params,
            target_device=device,
            keep_params=False,
            offload_cpu_strategy=offload_cpu_strategy,
        )

        lib = compile_nnp_model(
            mod, None, export_lib_path, target_host, target_host_cc, target_device
        )

    return lib


def compile_cpuref_model(
    mod,
    params,
    export_lib_path="",
    opt_level=2,
    target_host="llvm",
    target_host_cc=None,
    span_filter=None,
    config=None,
):
    """
    Compile cpuref relay IRModule.

    Parameters
    ==========
    mod: tvm.ir.module.IRModule
        The object of IRModule.

    params: dict of str to tvm.NDArray.array
        The params of IRModule.

    export_lib_path: str or list of str.
        If specified, export compiled model to path.

    opt_level: int
        The optimization level.

    target_host: list of str or tvm.target.Target
        Host targets specification, None for default host.

    target_host_cc: list of str
        Host targets cc toolchain path, None for default host cc.

    span_filter: set of str or func
        If specified as set, only preserve the span name in the set
        If specified as func, only preserve the span if func(span name) returns True

    config: dict
        The config of PassContext.

    Returns
    =======
    lib: tvm.runtime.Module or List[tvm.runtime.Module]
        Tvm graph runtime libs for each host.
    """

    def __arg_to_list(arg):
        if isinstance(arg, (list, tuple)):
            return arg
        return [arg]

    from tvm.contrib.edgex.relay.transform import ConvertCpuRefOutput

    # preserve span debug information
    mod = ConvertCpuRefOutput(span_filter)(mod)

    target_host_list = __arg_to_list(target_host)
    target_host_cc_list = __arg_to_list(target_host_cc)
    export_lib_path_list = __arg_to_list(export_lib_path)
    host_num = len(target_host_list)
    libs = []
    disabled_pass = [
        "SimplifyInference",
        "SimplifyExpr",
        "SimplifyExprPostAlterOp",
    ]
    with tvm.transform.PassContext(opt_level=opt_level, config=config, disabled_pass=disabled_pass):
        for i in range(host_num):
            target_host = target_host_list[i]
            target_host_cc = target_host_cc_list[i] if i < len(target_host_cc_list) else None
            export_lib_path = export_lib_path_list[i] if i < len(export_lib_path_list) else None
            if not isinstance(target_host, (str, tvm.target.Target, type(None))):
                raise RuntimeError("target_host should be string or tvm.target.Target or None.")
            if not isinstance(target_host_cc, (str, type(None))):
                raise RuntimeError("target_host_cc should be string or None.")
            if not isinstance(export_lib_path, (str, type(None))):
                raise RuntimeError("export_lib_path should be a string.")
            graph_lib = tvm.relay.build(mod, params=params, target=target_host)
            if export_lib_path is not None and export_lib_path != "":
                export_graph_library(
                    graph_lib, export_lib_path, target_host_cc=target_host_cc, ir_mod=mod
                )
            libs.append(graph_lib)

    return libs[0] if host_num == 1 else libs


def load_relay_module(mod_path: str, params_path: str = None):
    """Load relay module and params utility.

    Parameters
    ==========
    mod_path: str
        The path of relay IRModule file.
    params_path: str
        The path of params file.

    Returns
    =======
    mod: tvm.ir.IRModule
        tvm relay graph module.
    params: dict
        tvm param dict if params_path specified.
    """

    if not os.path.exists(mod_path):
        raise RuntimeError(f"Module path {mod_path} not exists.")
    with open(mod_path, "r") as infile:
        mod = tvm.ir.load_json(json.load(infile))
    params = None
    if params_path:
        if not os.path.exists(params_path):
            raise RuntimeError(f"Params path {params_path} not exists.")
        with open(params_path, "rb") as infile:
            params = dict(tvm.relay.load_param_dict(infile.read()))
    return mod, params


def save_relay_module(
    model_name: str, output_dir: str, mod: tvm.ir.IRModule, params: dict = None
) -> None:
    """Save relay module and params utility.

    Parameters
    ==========
    model_name: str
        output file name prefix.
    output_dir: str
        output directory.
    mod: tvm.ir.IRModule
        tvm relay graph module.
    params: dict
        tvm param dict.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"{model_name}.json"), "w") as outfile:
        json.dump(tvm.ir.save_json(mod), outfile)
    if params is None:
        params = dict()
    with open(os.path.join(output_dir, f"{model_name}.params"), "wb") as outfile:
        outfile.write(tvm.runtime.save_param_dict(params))


def run_and_check_iss(mod, params, lib):
    """
    Run and test the model.

    Parameters
    ==========
    mod: tvm.ir.module.IRModule
        The object of IRModule.
    params: dict of str to tvm.NDArray.array
        The params of IRModule.
    lib: tvm.Module
        Tvm lib file.

    Returns
    =======
    """

    if not isinstance(mod, tvm.ir.module.IRModule):
        raise RuntimeError("mod should be a tvm.ir.module.IRModule.")
    if not isinstance(params, (dict, type(None))):
        raise RuntimeError("params should be a dict or None.")
    if not isinstance(lib, tvm.relay.backend.executor_factory.ExecutorFactoryModule):
        raise RuntimeError(
            "lib should be a tvm.relay.backend.executor_factory.ExecutorFactoryModule."
        )

    from tvm.contrib.edgex.testing import check_edgex_relay_build

    check_edgex_relay_build(mod, params, check_cpu=True, rmse=0.001, edgex_lib=lib)


def get_available_graph_spans(lib):
    """Get all available span info from compiled library

    Parameters
    ==========
    lib: tvm.runtime.Module
        Compiled graph library

    Returns
    =======
    res: list
        A list of graph span infos in execution topological order.
        Each item is a dict like object of
        {
            "name": (span name)
            "func_name": (kernel name the span belongs to)
            "output_idx": (list of index into the kernel outputs)
            "node_idx": (graph json node id the span belongs to)
        }
    """
    results = []
    if isinstance(lib, relay.backend.executor_factory.GraphExecutorFactoryModule) or (
        isinstance(lib, tvm.runtime.Module) and lib.type_key in ("GraphExecutorFactory", "rpc")
    ):
        graph_json = json.loads(lib["get_graph_json"]())
    elif isinstance(lib, tvm.contrib.edgex.runtime.edgex_graph_executor.GraphModuleEdgexGraph):
        graph_json = lib._graph_json
    elif isinstance(lib, tvm.runtime.Module):
        raise ValueError(f"Unknown module type {lib.type_key}")
    else:
        raise ValueError(f"Unknown module type {type(lib)}")

    for idx, node in enumerate(graph_json["nodes"]):
        if node["op"] == "tvm_op":
            func_name = node["attrs"]["func_name"]
            if "LoweredFunctionNameHint" in node["attrs"]:
                func_name = node["attrs"]["LoweredFunctionNameHint"]
            if "span_info" in node["attrs"]:
                span_info = node["attrs"]["span_info"]
                try:
                    span_info = json.loads(base64.b64decode(span_info))
                    for span_name, out_idx in span_info.items():
                        span_dict = {}
                        span_dict["name"] = span_name
                        span_dict["func_name"] = func_name
                        span_dict["output_idx"] = out_idx
                        span_dict["node_idx"] = idx
                        results.append(span_dict)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.warning(f"Illegal span info: {span_info}:\n{e}")

    return results


def iss_layerwise_input_output(inputs, edgex_lib_or_path):
    """
    Run and get iss layerwise input and outputs.
    Parameters
    ----------
    inputs : [numpy.ndarray]
        List of numpy.ndarray.

    edgex_lib_or_path : tvm.relay.Module or str
        The edgex library or path to the library file.

    Returns
    -------
    layerwise_inputs: {func_name: [numpy.ndarray]}
        dict of layerwise inputs

    layerwise_outputs: {func_name: [numpy.ndarray]}
        dict of layerwise outputs of edgex
    """
    from tvm.contrib.edgex.relay.analysis import LayerwiseError

    if isinstance(edgex_lib_or_path, tvm.runtime.Module):
        lib = edgex_lib_or_path
    elif isinstance(edgex_lib_or_path, str):
        lib = tvm.runtime.load_module(edgex_lib_or_path)
    else:
        raise ValueError("edgex_lib_or_path should be a tvm.runtime.Module or str.")

    layerwise_outputs, layerwise_inputs, _, _ = LayerwiseError.run(lib, inputs)
    return layerwise_inputs, layerwise_outputs


# def custom_func_build(
#     inputs: Union[PrimFunc, Callable],
#     args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
#     target: Optional[Union[str, Target]] = None,
#     name: Optional[str] = "default_function",
#     poly_opt: int = 0,
# ):
#     """Analyze remap info to build block, call naive schedule and build

#     Parameters
#     ==========
#     inputs : PrimFunc
#         An user-defined function.

#     args : Optional[List[Union[tvm.tir.Buffer, tensor.Tensor, Var]]]
#         The argument lists to the function.

#     target : Optional[Union[str, Target]]
#         The target and option of the compilation.

#     name : Optional[str]
#         The name of result function.

#     Returns
#     ==========
#     ret : tvm.module
#         A module that combines both host and device code.
#     """
#     if isinstance(inputs, Callable):
#         if getattr(inputs, "edgex_op"):
#             if getattr(inputs, "debug"):
#                 # return directly to run and debug with python interpreter
#                 return inputs
#         else:
#             raise ValueError("The input function is not an edgex op")
#     if poly_opt == 1:
#         func_with_block = OptAndBuildBlock()(IRModule({"main": inputs}))["main"]
#     elif poly_opt == 2:
#         func_with_block = PolyBuildBlock()(IRModule({"main": inputs}))["main"]
#     else:
#         func_with_block = AnalyzeIteratorDomain()(IRModule({"main": inputs}))["main"]
#     with build_config_nnp():
#         func_afer_sch = tvm.contrib.edgex.topi.naive_vu_schedule(func_with_block)
#         func_after_build = tvm.build(func_afer_sch, args, target, name=name)
#     return func_after_build


def estimate_origin_mod_FLOPs(mod):
    """
    Estimate the origin model's FLOPs.

    Parameters
    ==========
    mod: tvm.ir.module.IRModule
        The object of IRModule.

    Returns
    =======
    result: dict
       A dict of FLOPs for each operator, separated by different data types.
    """

    if not isinstance(mod, tvm.ir.module.IRModule):
        raise RuntimeError("mod should be a tvm.ir.module.IRModule.")

    from tvm.contrib.edgex.relay.analysis import count_flops_on_module

    result = count_flops_on_module(mod)
    return result


def estimate_compiled_mod_Cycles(mod):
    """
    Estimate the compiled model's cycles.

    Parameters
    ==========
    mod: tvm.runtime.Module
        The object of Module.

    Returns
    =======
    result: dict
       A dict of string to cycles for each function.
    """

    if not isinstance(mod, tvm.runtime.Module):
        raise RuntimeError("mod should be a tvm.runtime.Module.")

    result = {}
    for node in json.loads(mod["get_graph_json"]())["nodes"]:
        if node["op"] == "tvm_op":
            func_name = node["attrs"]["func_name"]
            if "nnp_cycles" in node["attrs"]:
                result[func_name] = int(node["attrs"]["nnp_cycles"])
            else:
                result[func_name] = 0
                logger.warning(f"nnp_cycles is not found in {func_name}")
    return result


def estimate_compiled_mod_MACs(mod):
    """
    Estimate the compiled model's MACs.

    Parameters
    ==========
    mod: tvm.runtime.Module
        The object of Module.

    Returns
    =======
    result: dict
       A dict of string to MACs for each function.
    """

    if not isinstance(mod, tvm.runtime.Module):
        raise RuntimeError("mod should be a tvm.runtime.Module.")

    result = {}
    for node in json.loads(mod["get_graph_json"]())["nodes"]:
        if node["op"] == "tvm_op":
            func_name = node["attrs"]["func_name"]
            if "nnp_macs" in node["attrs"]:
                result[func_name] = int(node["attrs"]["nnp_macs"])
            else:
                result[func_name] = 0
                logger.warning(f"nnp_macs is not found in {func_name}")
    return result


def estimate_compiled_mod_vcu_cycles(mod):
    """
    Estimate the compiled model's vu cycles.

    Parameters
    ==========
    mod: tvm.runtime.Module
        The object of Module.

    Returns
    =======
    result: dict
       A dict of string to vu cycles for each function.
    """

    if not isinstance(mod, tvm.runtime.Module):
        raise RuntimeError("mod should be a tvm.runtime.Module.")

    result = {}
    for node in json.loads(mod["get_graph_json"]())["nodes"]:
        if node["op"] == "tvm_op":
            func_name = node["attrs"]["func_name"]
            if "nnp_vcu_cycles" in node["attrs"]:
                result[func_name] = int(node["attrs"]["nnp_vcu_cycles"])
            else:
                result[func_name] = 0
                logger.warning(f"nnp_vcu_cycles is not found in {func_name}")
    return result


def get_version():
    """
    Return the nnp toolkits version.

    Parameters
    ==========

    Returns
    =======
    result: dict
       Version info dict for tytvm components.
    """
    from tvm.contrib.edgex.config import get_edgex_version_info

    return get_edgex_version_info()


def save(mod, params=None, output_path="./output.onnx"):
    """Save reay ir in onnx format.

    Parameters
    ==========
    mod: str or IRModule
        The mod or the path of json.
    params_path: str or dict
        The dict or the path of params.
    output_path: str
        The file path to save onnx file.

    Returns
    =======
    result: Models in onnx format
    """
    from .utils.export_relay_to_onnx import RelayExporter

    return RelayExporter().save(mod, params=params, output_path=output_path)


def load(file_path: str):
    """Load relay ir in onnx format

    Parameters
    ==========
    file_path: str
        The file path of onnx.

    Returns
    =======
    mod: tvm.ir.IRModule
        tvm relay graph module.
    params: dict
        tvm param dict.
    """
    from .utils.convert_onnx2relay import OnnxExporter

    onnx_model = onnx.load(file_path)
    graph = onnx_model.graph
    mod, params = OnnxExporter().from_onnx(graph)
    return mod, params
