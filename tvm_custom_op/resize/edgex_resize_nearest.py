# encoding=utf-8
from tvm.contrib.edgex.relay import CUSTOM_OP, CustomCpuExternOp


# 继承 CustomCpuExternOp，实现到外部函数的绑定
@CUSTOM_OP.register("EdgexResizeNearest")
class EdgexResizeNearestOp(CustomCpuExternOp):

    # 提供算子的外部C函数名
    func_name = "EdgexResizeNearest_arm"

    # 提供源代码或者链接库路径，如果为C代码文件则自动编译
    files = ["edgex_resize_nearest.cc"]

    # 实现算子类型推断
    def infer_type(self, inputs, attrs):
        in_shape = inputs[0]["shape"]
        in_dtype = inputs[0]["dtype"]
        return [{"shape": [1, 3, 2, 2], "dtype": "float32"}]

    # 支持传入额外参数
    # c函数调用约定: 输入0，输入1，..., 输出0，输出1，..., 额外标量参数
    def gen_extra_args(self, inputs, attrs, outputs):
        # 这里我们额外传入图片尺寸，shape参数也可以从入参tensor结构体中获取
        # C函数签名为 void f(DLTensor* x, DLTensor* y, int N, int C, int H, int W)
        # 如果不需要额外参数，可不定义或返回None
        scale_factor = attrs["scale_factor"]
        return [scale_factor]

