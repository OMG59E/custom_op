import tvm
import tvm.contrib.graph_executor as graph_executor
import numpy as np


lib = tvm.runtime.load_module("model.x86.O2.so")
engine = graph_executor.GraphModule(lib["default"](tvm.edgex(), tvm.cpu()))

x = np.random.random((1, 3, 4, 4)).astype(np.float32)
in_datas = {"x": x}
engine.set_input(**in_datas)
engine.run()
outputs = list()
for idx in range(engine.get_num_outputs()):
    outputs.append(engine.get_output(idx).asnumpy())
print(outputs)
