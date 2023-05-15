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
        a.array, out.array, n_elements, tshape, tstrides, d, offset, BLOCK_SIZE=128)
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

    # TODO: BLOCK_SIZE > 2 would still cause error
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    scalar_setitem_kernel[grid](
        val, out.array, n_elements, tshape, tstrides, d, offset, BLOCK_SIZE=1
    )
    return out

op_enum = {
    "add": 0,
    "mul": 1,
    "div": 2,
    "eq": 3,
    "ge": 4,
    "log": 5,
    "exp": 6,
    "tanh": 7,
    "power": 8,
    "maximum": 9,
}

@triton.jit
def ewise_unary_op_kernel(
    a_ptr,
    out_ptr,
    n_elements,
    op_code,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)

    # NOTE: strict grammar on if/else, and output initialization
    if op_code == 5:
        output = tl.log(a)
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 6:
        output = tl.exp(a)
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 7:
        output = (tl.exp(a) - tl.exp(-a)) / (tl.exp(a) + tl.exp(-a))
        tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def ewise_binary_op_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    op_code,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # NOTE: strict grammar on if/else, and output initialization
    if op_code == 0:
        output = a + b
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 1:
        output = a * b
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 2:
        output = a / b
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 3:
        output = a == b
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 4:
        output = a > b
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 9:
        output = tl.maximum(a, b)
        tl.store(out_ptr + offsets, output, mask=mask)

@triton.jit
def scalar_binary_op_kernel(
    a_ptr,
    val,
    out_ptr,
    n_elements,
    op_code,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)

    # NOTE: strict grammar on if/else, and output initialization
    if op_code == 0:
        output = a + val
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 1:
        output = a * val
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 2:
        output = a / val
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 3:
        output = a == val
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 4:
        output = a > val
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 8:
        output = a ** val
        tl.store(out_ptr + offsets, output, mask=mask)
    elif op_code == 9:
        output = tl.maximum(a, val)
        tl.store(out_ptr + offsets, output, mask=mask)

def ewise_binary_op(a, b, out, op_name):
    assert a.array.is_cuda and b.array.is_cuda and out.array.is_cuda
    n_elements = out.array.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    ewise_binary_op_kernel[grid](a.array, b.array, out.array, n_elements, op_enum[op_name], BLOCK_SIZE=1024)
    return out

def ewise_unary_op(a, out, op_name):
    assert a.array.is_cuda and out.array.is_cuda
    n_elements = out.array.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    ewise_unary_op_kernel[grid](a.array, out.array, n_elements, op_enum[op_name], BLOCK_SIZE=1024)
    return out

def scalar_binary_op(a, val, out, op_name):
    assert a.array.is_cuda and out.array.is_cuda
    n_elements = out.array.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    scalar_binary_op_kernel[grid](a.array, val, out.array, n_elements, op_enum[op_name], BLOCK_SIZE=1024)
    return out

def ewise_add(a, b, out):
    return ewise_binary_op(a, b, out, "add")


def scalar_add(a, val, out):
    return scalar_binary_op(a, val, out, "add")


def ewise_mul(a, b, out):
    return ewise_binary_op(a, b, out, "mul")


def scalar_mul(a, val, out):
    return scalar_binary_op(a, val, out, "mul")


def ewise_div(a, b, out):
    return ewise_binary_op(a, b, out, "div")


def scalar_div(a, val, out):
    return scalar_binary_op(a, val, out, "div")


def scalar_power(a, val, out):
    return scalar_binary_op(a, val, out, "power")


def ewise_maximum(a, b, out):
    return ewise_binary_op(a, b, out, "maximum")


def scalar_maximum(a, val, out):
    return scalar_binary_op(a, val, out, "maximum")


def ewise_eq(a, b, out):
    return ewise_binary_op(a, b, out, "eq")


def scalar_eq(a, val, out):
    return scalar_binary_op(a, val, out, "eq")


def ewise_ge(a, b, out):
    return ewise_binary_op(a, b, out, "ge")


def scalar_ge(a, val, out):
    return scalar_binary_op(a, val, out, "get")


def ewise_log(a, out):
    return ewise_unary_op(a, out, "log")


def ewise_exp(a, out):
    return ewise_unary_op(a, out, "exp")


def ewise_tanh(a, out):
    return ewise_unary_op(a, out, "tanh")


def matmul(a, b, out, m, n, p):
    raise NotImplementedError


def reduce_max(a, out, reduce_size):
    raise NotImplementedError


def reduce_sum(a, out, reduce_size):
    raise NotImplementedError


# %%
