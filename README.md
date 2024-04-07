# PyTorch/onnxruntime/tvm custom op

## requirements
- python==3.8
- torch==2.2.2+cpu
- onnx
- eigen3
- onnxruntime==1.13.1


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
python build.py
```