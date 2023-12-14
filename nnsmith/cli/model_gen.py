import logging
import os
import random
import time

import hydra
from omegaconf import DictConfig

from nnsmith.abstract.extension import activate_ext
from nnsmith.backends.factory import BackendFactory
from nnsmith.graph_gen import SymbolicGen, model_gen, viz
from nnsmith.logging import MGEN_LOG
from nnsmith.materialize import Model, TestCase
from nnsmith.narrow_spec import auto_opset
from nnsmith.util import hijack_patch_requires, mkdir, op_filter


@hydra.main(version_base=None, config_path="../config", config_name="main")  # 利用hydra去加载一些配置信息
def main(cfg: DictConfig):
    # Generate a random ONNX model
    # TODO(@ganler): clean terminal outputs.
    mgen_cfg = cfg["mgen"]  # 加载配置项

    seed = random.getrandbits(32) if mgen_cfg["seed"] is None else mgen_cfg["seed"]  # 初始化随机种子

    MGEN_LOG.info(f"Using seed {seed}")

    # TODO(@ganler): skip operators outside of model gen with `cfg[exclude]`
    model_cfg = cfg["model"]
    ModelType = Model.init(model_cfg["type"], backend_target=cfg["backend"]["target"])  # 初始化模型类型
    ModelType.add_seed_setter()  # 给这个模型配置上随机种子

    if cfg["backend"]["type"] is not None:
        factory = BackendFactory.init(  # 初始化后端工厂
            cfg["backend"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
            parse_name=True,
        )
    else:
        factory = None

    # GENERATION
    opset = auto_opset(  # 根据模型类型和后端工厂，返回一个算子集
        ModelType,
        factory,
        vulops=mgen_cfg["vulops"],
        grad=mgen_cfg["grad_check"],
    )
    opset = op_filter(opset, mgen_cfg["include"], mgen_cfg["exclude"])  # 对算子集做一个初步的过滤
    hijack_patch_requires(mgen_cfg["patch_requires"])  # 动态配置程序文件
    activate_ext(opset=opset, factory=factory)  # 激活特定的扩展

    tgen_begin = time.time()  # 模型开始生成的时间
    gen = model_gen(  # 实例化一个模型生成器
        opset=opset,
        method=mgen_cfg["method"],
        seed=seed,
        max_elem_per_tensor=mgen_cfg["max_elem_per_tensor"],
        max_nodes=mgen_cfg["max_nodes"],
        timeout_ms=mgen_cfg["timeout_ms"],
        rank_choices=mgen_cfg["rank_choices"],
        dtype_choices=mgen_cfg["dtype_choices"],
    )
    tgen = time.time() - tgen_begin

    if isinstance(gen, SymbolicGen):  # 如果生成的模型是符号生成模型（SymbolicGen 类的实例），则记录符号数和约束数，并在调试级别下记录解决方案。
        MGEN_LOG.info(
            f"{len(gen.last_solution)} symbols and {len(gen.solver.assertions())} constraints."
        )

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug("solution:" + ", ".join(map(str, gen.last_solution)))

    # 实例化一个GraphIR
    tmat_begin = time.time()
    ir = gen.make_concrete()

    MGEN_LOG.info(
        f"Generated DNN has {ir.n_var()} variables and {ir.n_compute_inst()} operators."
    )

    mkdir(mgen_cfg["save"])
    if cfg["debug"]["viz"]:
        fmt = cfg["debug"]["viz_fmt"].replace(".", "")
        viz(ir, os.path.join(mgen_cfg["save"], f"graph.{fmt}"))

    model = ModelType.from_gir(ir)  # 创建一个模型实例
    model.refine_weights()  # 优化权重
    model.set_grad_check(mgen_cfg["grad_check"])  # 设置梯度检查
    oracle = model.make_oracle()  # 创建一个模型对应的oracle实例
    tmat = time.time() - tmat_begin

    tsave_begin = time.time()
    testcase = TestCase(model, oracle)  # 将模型和oracle包装成一个testcase
    testcase.dump(root_folder=mgen_cfg["save"])  # 将这套测试用例保存下来
    tsave = time.time() - tsave_begin

    MGEN_LOG.info(
        f"Time:  @Generation: {tgen:.2f}s  @Materialization: {tmat:.2f}s  @Save: {tsave:.2f}s"
    )


if __name__ == "__main__":
    main()
