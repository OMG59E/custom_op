# PyTorch/onnxruntime/tvm custom op

## requirements
- python==3.8
- torch==2.2.2+cpu
- onnx
- eigen3
- onnxruntime==1.13.1
- tytvm==0.8.0


## PyTorch

```bash
# step1 build
python setup.py build
# step2 export onnx
python export_custom_op.py
# step3 run
python run_infer.py
```

## onnxruntime

```bash
mkdir build && cd build
cmake .. && make -j4
./customop  # c++
python run_infer.py  # python
```

## tvm

```bash
# 设置ARM_C_COMPILER
source env.sh

# 依赖三方库的情况下，通过修改edgex.py 857行export_graph_library函数，添加header和lib
mv ${TVM_HOME}/contrib/edgex/edgex.cpython-38-x86_64-linux-gnu.so .
cp edgex.py ${TVM_HOME}/contrib/edgex/

# 目前080量化有问题，需要替换config.py
mv ${TVM_HOME}/relay/quantization/config.cpython-38-x86_64-linux-gnu.so .
cp config.py ${TVM_HOME}/relay/quantization/

# 编译
python build.py
```