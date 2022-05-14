import time
from typing import List, Tuple
from abc import ABC, abstractmethod

import torch
import numpy as np

from nnsmith.graph_gen import SymbolNet, random_tensor


class InputSearchBase(ABC):
    @staticmethod
    def apply_weights(net, weight_sample):
        with torch.no_grad():
            for name, param in net.named_parameters():
                param.copy_(weight_sample[name])

    def __init__(self, net: SymbolNet, start_inputs=None, start_weights=None, use_cuda=False):
        self.net = net
        self.start_inputs = start_inputs
        self.start_weights = start_weights
        self.use_cuda = use_cuda

    @abstractmethod
    def search_one(self, start_inp, timeout_ms: int = None) -> List[torch.Tensor]:
        pass

    def search(self, max_time_ms: int = None, max_sample: int = 1) -> Tuple[int, Tuple[str, np.ndarray]]:
        n_try = 0
        sat_inputs = None
        start_time = time.time()

        while (max_time_ms is None or time.time() - start_time < max_time_ms / 1000) and n_try < max_sample:
            if self.start_weights is not None and n_try < len(self.start_weights):
                self.apply_weights(self.net, self.start_weights[n_try])
            else:
                weight_sample = {}
                with torch.no_grad():
                    for name, param in self.net.named_parameters():
                        weight_sample[name] = random_tensor(
                            param.shape, dtype=param.dtype, use_cuda=self.use_cuda)
                self.apply_weights(self.net, weight_sample)

            if self.start_inputs is not None and n_try < len(self.start_inputs):
                cur_input = self.start_inputs[n_try]
            else:
                cur_input = self.net.get_random_inps(use_cuda=self.use_cuda)

            res = self.search_one(cur_input, max_time_ms)
            if res is not None:
                sat_inputs = res
                break
            n_try += 1

        if sat_inputs is not None:
            sat_inputs = {name: inp for name,
                          inp in zip(self.net.input_spec, sat_inputs)}

        return n_try, sat_inputs


class SamplingSearch(InputSearchBase):
    # Think about how people trivially generate inputs.
    def search_one(self, start_inp, timeout_ms: int = None) -> List[torch.Tensor]:
        with torch.no_grad():
            self.net.check_intermediate_numeric = True
            _ = self.net(*start_inp)
            if not self.net.invalid_found_last:
                return start_inp

            return None


class GradSearch(InputSearchBase):
    def search_one(self, start_inp, timeout_ms: int = None) -> List[torch.Tensor]:
        return self.net.grad_input_gen(
            init_tensors=start_inp, use_cuda=self.use_cuda, max_time=timeout_ms / 1000)
