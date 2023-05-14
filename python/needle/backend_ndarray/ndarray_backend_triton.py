import torch

import triton
import triton.language as tl

__device_name__ = "triton"
_datatype = torch.float32
_datetype_size = torch.tensor([], dtype=torch.float32).element_size()


class Array:
    """1D array backed by torch.Tensor.

    The heavy lifting is done via @triton.jit method.
    """
    def __init__(self, size):
        self.array = torch.empty(size, dtype=torch.float32, device="cuda")

    @property
    def size(self):
        return self.array.size


def to_numpy(a, shape, strides, offset):
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple([s * _datetype_size for s in strides])
    )

def from_numpy(a, out):
    out.array = torch.from_numpy(a.flatten())


def fill(out, val):
    out.array.fill(val)


def compact(a, out, shape, strides, offset):
    raise NotImplementedError


def ewise_setitem(a, out, shape, strides, offset):
    raise NotImplementedError


def scalar_setitem(size, val, out, shape, strides, offset):
    raise NotImplementedError


def ewise_add(a, b, out):
    raise NotImplementedError


def scalar_add(a, val, out):
    raise NotImplementedError


def ewise_mul(a, b, out):
    raise NotImplementedError


def scalar_mul(a, val, out):
    raise NotImplementedError


def ewise_div(a, b, out):
    raise NotImplementedError


def scalar_div(a, val, out):
    raise NotImplementedError


def scalar_power(a, val, out):
    raise NotImplementedError


def ewise_maximum(a, b, out):
    raise NotImplementedError


def scalar_maximum(a, val, out):
    raise NotImplementedError


def ewise_eq(a, b, out):
    raise NotImplementedError


def scalar_eq(a, val, out):
    raise NotImplementedError


def ewise_ge(a, b, out):
    raise NotImplementedError


def scalar_ge(a, val, out):
    raise NotImplementedError


def ewise_log(a, out):
    raise NotImplementedError


def ewise_exp(a, out):
    raise NotImplementedError


def ewise_tanh(a, out):
    raise NotImplementedError


def matmul(a, b, out, m, n, p):
    raise NotImplementedError


def reduce_max(a, out, reduce_size):
    raise NotImplementedError


def reduce_sum(a, out, reduce_size):
    raise NotImplementedError
# %%
