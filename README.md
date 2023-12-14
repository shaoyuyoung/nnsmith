# NNSmith Source Code
## experiment
experiment中主要包含了一些基本的用于实验的代码，这里主要是用来计算覆盖率的
* `evaluate_models.py`: 用于执行生成好的模型，使用llvm生成`*.profraws`文件，用于后续计算测试覆盖率
* `process_profraws.py`: 处理`.profraws`文件，获得测试覆盖率的信息
* `viz_merged_cov.py`: 可视化结果，生成图标

## nnsmith
这是nnsmith工具的主要源码部分
* abstract
<br>本系统所定义的内部各类抽象定义，主要为抽象数据类型、抽象算子、抽象张量
  * `arith.py`
  * `dtype.py`
  * `extension.py`
  * `op.py`
  * `tensor.py`
* backends
  * `factory.py`
  * `onnxruntime.py`
  * `pt2.py`
  * `tensorrt.py`
  * `tflite.py`
  * `torchjit.py`
  * `tvm.py`
  * `xla.py`
* cli
<br>项目脚手架，用于fuzzing和difftesting
  * `dtype_test.py`: 数据类型测试
  * `fuzz.py`: 模糊测试程序入口
  * `model_exec.py`: 用于执行模型，在计算测试覆盖率阶段会用到
  * `model_gen.py`: 用于生成模型
* config
* materialize
<br>定义了前端框架的一些基本行为，使GraphIR具体化为某一个框架的模型
* `difftest.py`
* `error.py`
* `filter.py`
* `gir.py`
* `graph_gen.py`: 核心模块
* `logging.py`
* `macro.py`
* `narrow_spec.py`
* `util.py`

## requirements
包含了测试各个系统的requirements
