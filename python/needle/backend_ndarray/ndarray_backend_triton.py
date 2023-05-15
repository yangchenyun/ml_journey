import numpy as np
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
        self.array = torch.empty(size, device="cuda", dtype=torch.float32)

    @property
    def size(self):
        assert len(self.array.shape) == 1
        return self.array.size()[0]

    def ptr(self):
        return self.array.data_ptr()


def to_numpy(a, shape, strides, offset):
    return np.lib.stride_tricks.as_strided(
        a.array[offset:].cpu(), shape, tuple([s * _datetype_size for s in strides])
    )


def from_numpy(a, out):
    out.array = torch.from_numpy(a.flatten().astype("float32")).to(out.array.device)
    assert out.array.is_cuda
    assert out.array.dtype == _datatype


def fill(out, val):
    out.array.fill(val)


@triton.jit
def compact_stride_index(tid, n_elements, tshape_ptr, tstrides_ptr, d):
    # NOTE: weird variable name overloading for offset, (does the function lined?)
    local_offset = 0
    remaining_indices = tid
    for di in range(0, d):
        step_size = 1
        for j in range(di + 1, d):
            step_size *= tl.load(tshape_ptr + j)
        idx = remaining_indices // step_size  # use floor division to get int
        local_offset += idx * tl.load(tstrides_ptr + di)
        remaining_indices -= idx * step_size
    return local_offset


@triton.jit
def compact_kernel(
    a_ptr,
    out_ptr,
    n_elements,
    tshape_ptr,
    tstrides_ptr,
    d,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # NOTE: use pointer to access array data type
    # NOTE: Does each program runs in a single thread or multiple threads?
    for i in range(0, BLOCK_SIZE):
        tid = pid * BLOCK_SIZE + i
        if tid < n_elements:
            stride_i = offset + compact_stride_index(
                tid, n_elements, tshape_ptr, tstrides_ptr, d
            )
            a_val = tl.load(a_ptr + stride_i)
            tl.store(out_ptr + tid, a_val)


def compact(a, out, shape, strides, offset):
    assert out.array.is_cuda
    assert a.array.is_cuda
    assert len(shape) == len(strides)
    n_elements = out.array.numel()

    # NOTE: turn shape and strides into torch array to pass to kernel
    tshape = torch.tensor(shape, device="cuda", dtype=torch.int32)
    tstrides = torch.tensor(strides, device="cuda", dtype=torch.int32)
    d = len(shape)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    compact_kernel[grid](
        a.array, out.array, n_elements, tshape, tstrides, d, offset, BLOCK_SIZE=128
    )
    return out


@triton.jit
def ewise_setitem_kernel(
    a_ptr,
    out_ptr,
    n_elements,
    tshape_ptr,
    tstrides_ptr,
    d,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    for i in range(0, BLOCK_SIZE):
        tid = pid * BLOCK_SIZE + i
        if tid < n_elements:
            stride_i = offset + compact_stride_index(
                tid, n_elements, tshape_ptr, tstrides_ptr, d
            )
            a_val = tl.load(a_ptr + tid)
            tl.store(out_ptr + stride_i, a_val)


def ewise_setitem(a, out, shape, strides, offset):
    assert out.array.is_cuda
    assert a.array.is_cuda
    assert len(shape) == len(strides)

    # NOTE: be careful about the number of elements to process
    n_elements = a.array.numel()

    tshape = torch.tensor(shape, device="cuda", dtype=torch.int32)
    tstrides = torch.tensor(strides, device="cuda", dtype=torch.int32)
    d = len(shape)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    ewise_setitem_kernel[grid](
        a.array, out.array, n_elements, tshape, tstrides, d, offset, BLOCK_SIZE=128
    )
    return out


@triton.jit
def scalar_setitem_kernel(
    val,
    out_ptr,
    n_elements,
    tshape_ptr,
    tstrides_ptr,
    d,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # NOTE: use pointer to access array data type
    # NOTE: Does each program runs in a single thread or multiple threads?
    for i in range(0, BLOCK_SIZE):
        tid = pid * BLOCK_SIZE + i
        if tid >= n_elements:
            return
        stride_i = offset + compact_stride_index(
            tid, n_elements, tshape_ptr, tstrides_ptr, d
        )
        tl.store(out_ptr + stride_i, val)


def scalar_setitem(size, val, out, shape, strides, offset):
    assert out.array.is_cuda
    assert len(shape) == len(strides)
    n_elements = size
    tshape = torch.tensor(shape, device="cuda", dtype=torch.int32)
    tstrides = torch.tensor(strides, device="cuda", dtype=torch.int32)
    d = len(shape)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    scalar_setitem_kernel[grid](
        val, out.array, n_elements, tshape, tstrides, d, offset, BLOCK_SIZE=1
    )
    return out


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
