import os
import tvm
from tvm.contrib.edgex.utils import export_relay_to_onnx
from tvm.contrib.edgex import load_model_from_file
from tvm.contrib.edgex import get_version
from tvm.contrib.edgex import optimize_and_compile
from edgex_dropout import EdgexDropoutOp

print(get_version())

# onnx to relay_func
save_dir = "outputs"
onnx_file = "../../pytorch_custom_op/dropout/dropout_forward.onnx"
shape_dict = {"input": (1, 3, 3, 3)}
mod, params = load_model_from_file(onnx_file, "onnx", shape_dict)

# quantization
in_dtypes = {"input": "float32"}
quantize_config = tvm.relay.quantization.get_quantize_config("nnp400", in_dtypes)
quantize_config["calib_method"] = "kld"
quantize_config["level"] = 0
    
mod_quant, params_quant = tvm.relay.quantization.quantize(
    mod, 
    params, 
    model_name="opt_ir", 
    dataset=None,
    prof_img_num=1,
    quantize_config=quantize_config,
    debug_level=-1,
    save_dir=save_dir,
)


ARM_C_COMPILER = os.getenv("ARM_C_COMPILER")
assert os.path.exists(ARM_C_COMPILER), "Not found ARM_C_COMPILER env"
target_host = ["llvm -mtriple=x86_64", "llvm -mtriple=aarch64"]
target_host_cc = [None, ARM_C_COMPILER]
target_device = tvm.target.Target("edgex", host="edgex_virtual_host")
export_lib_path = ["model.x86.O2.so", "model.a55.O2.so"]

# export_lib_path = ["model.x86.O2.so"]
# target_host = ["llvm -mtriple=x86_64"]
# target_host_cc = [None]
# target_device = tvm.target.Target("edgex")

libs = optimize_and_compile(
    mod_quant,
    params_quant,
    working_dir=save_dir,
    export_lib_path=export_lib_path,
    opt_level=2,
    target_host=target_host,
    target_host_cc=target_host_cc,
    target_device=target_device,
    # config=config
)

