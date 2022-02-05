import random
import warnings
from nnsmith.backends import DiffTestBackend
import pickle
from pathlib import Path
from typing import List, Union, Dict, Tuple
import time
import numpy as np
from tqdm import tqdm
from nnsmith import difftest, util, input_gen


class CrashExecutor(DiffTestBackend):
    """For testing purposes"""

    def predict(self, model, inputs):
        assert False


class HangExecutor(DiffTestBackend):
    """For testing purposes"""

    def predict(self, model, inputs):
        while True:
            pass


class BackendCreator:
    NAME_MAP = {
        'ort': 'ORTExecutor',
        'tvm-llvm': 'TVMExecutorLLVM',
        'tvm-debug': 'TVMExecutorDebug',
        'tvm-cuda': 'TVMExecutor',
        'xla': 'XLAExecutor',
        'trt': 'TRTBackend',
    }

    def __init__(self, name):
        self.name = name
        self.dump_name = self.NAME_MAP[name]

    def __call__(self, *args, **kwargs):
        name = self.name
        if name == 'ort':
            print('importing ort....')
            from nnsmith.backends.ort_graph import ORTExecutor
            print('imported')
            return ORTExecutor()
        elif name == 'tvm-debug':
            from nnsmith.backends.tvm_graph import TVMExecutor
            return TVMExecutor(executor='debug', opt_level=0)
        elif name == 'tvm-llvm':
            from nnsmith.backends.tvm_graph import TVMExecutor
            return TVMExecutor(target='llvm')
        elif name == 'tvm-cuda':
            from nnsmith.backends.tvm_graph import TVMExecutor
            return TVMExecutor(target='cuda')
        elif name == 'xla':
            from nnsmith.backends.xla_graph import XLAExecutor
            return XLAExecutor(device='CUDA')
        elif name == 'trt':
            from nnsmith.backends.trt_graph import TRTBackend
            return TRTBackend()
        elif name == 'crash':
            return CrashExecutor()
        elif name == 'hang':
            return HangExecutor()
        else:
            raise ValueError(f'unknown backend: {name}')


def run_backend_single_model(model_path: str, domain_path: str, backend: BackendCreator, dump_raw: str, seed: int = None, infer_domain=True):
    """This function is for debugging purpose.
    Run the backend on the same process.
    Compared to run_backend_single_model_raw_input, this is new version with input gen on the fly
    """
    backend = backend()
    model = DiffTestBackend.get_onnx_proto(model_path)
    inp_spec = DiffTestBackend.analyze_onnx_io(model)[0]
    if domain_path is not None:
        rngs = pickle.load(open(domain_path, 'rb'))
    elif infer_domain:
        rngs = input_gen.InputGenV3().infer_domain(model)
        print('inferred domain', rngs)
    else:
        rngs = None
    inputs = input_gen.gen_one_input_rngs(
        inp_spec, rngs, seed)
    outputs = backend.predict(model_path, inputs)
    if dump_raw is not None:
        pickle.dump(inputs, open(dump_raw + ".input", 'wb'))
        pickle.dump(outputs, open(dump_raw + ".output", 'wb'))
    return outputs


def run_backend_single_model_raw_input(model_path: str, input_path: str, backend: BackendCreator, dump_raw: str):
    """This function is for debugging purpose.
    Run the backend on the same process. gen_input is the index of the input to generate.
    """
    backend = backend()
    if input_path is not None:
        inputs = pickle.load(Path(input_path).open('rb'))
        outputs = backend.predict(model_path, inputs)
    else:
        outputs = []
        for inp_path in tqdm(sorted(list(Path(model_path).parent.glob(f'input.*.pkl')))):
            inputs = pickle.load(inp_path.open('rb'))
            outputs.append(backend.predict(model_path, inputs))
    if dump_raw is not None:
        pickle.dump(outputs, open(dump_raw, 'wb'))
    return outputs


def summarize(outputs: Dict[str, np.ndarray]):
    m = {k + '_mean': np.mean(o) for k, o in outputs.items()}
    # TODO(JK): figure out how to deal with nan
    m.update({k + '_nanmean': np.nanmean(o) for k, o in outputs.items()})
    m.update({k + '_num_nan': np.sum(np.isnan(o)) for k, o in outputs.items()})
    return m


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, required=True,
                        help=f'One of {BackendCreator.NAME_MAP.keys()}')
    parser.add_argument('--model', type=str,
                        help='For debugging purpose: path to onnx model;')
    parser.add_argument('--input_domain', type=str,
                        help='When gen_input is specified: path to the domain pkl file. If not specified, it uses the default domain.')
    parser.add_argument(
        '--dump_raw', help='Dumps the raw output to the specified path')
    parser.add_argument('--raw_input', type=str,
                        help='When specified, the model will be fed with the specified input. Otherwise, input will be generated on the fly.')
    parser.add_argument('--seed', type=int,
                        help='to generate random input data')
    parser.add_argument('--cmp_with', type=str, default=None,
                        help='the backend to compare with')
    parser.add_argument('--no_infer', action='store_true',
                        help='Use default domain instead of inferring')

    # TODO: Add support for passing backend-specific options
    args = parser.parse_args()
    if args.raw_input is not None:
        assert args.input_domain is None, \
            f"`input_domain`={args.input_domain} should not be specified when `raw_input` is specified." + \
            "`raw_input` mode is mutually exclusive with `gen-on-the-fly` mode."

    st = time.time()
    if args.seed is None:
        seed = random.getrandbits(32)
    else:
        seed = args.seed
    print('Using seed:', seed)

    def run_backend(bknd, dump_raw):
        if args.raw_input is not None:
            return run_backend_single_model_raw_input(args.model, args.raw_input, bknd, dump_raw)
        else:
            return run_backend_single_model(args.model, args.input_domain, bknd, dump_raw, seed, not args.no_infer)

    outputs = run_backend(BackendCreator(args.backend), args.dump_raw)
    if args.cmp_with is not None:
        oracle = BackendCreator(args.cmp_with)
        outputs_oracle = run_backend(
            oracle, None if args.dump_raw is None else args.dump_raw + ".oracle")
        difftest.assert_allclose(
            outputs, outputs_oracle, args.backend, args.cmp_with)
    print(f'Total time: {time.time() - st}')
