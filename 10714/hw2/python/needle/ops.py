"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy
import functools
import needle

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)

class ArgMax(TensorOp):
    def __init__(self, axis: Optional[int] = None):
        self.axis = axis

    def compute(self, a: NDArray):
        return array_api.argmax(a, axis=self.axis)

    def gradient(self, out_grad: Tensor, node: Tensor):
        raise NotImplementedError("Argmax gradient not implemented yet.")


def argmax(a, axis=None):
    return ArgMax(axis)(a)


class Max(TensorOp):
    """ Not tested."""
    def __init__(self, axes: Optional[int] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        return array_api.max(a, axes=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        ai = argmax(a, self.axes)
        da = array_like.zeros_like(a.shape)
        da[ai] = 1  # Only the max element has gradient 1
        da = needle.Tensor(da)
        return (out_grad * da,)


def max(a, axes=None):
    return Max(axes)(a)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)

class EWiseNe(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a != b

    def gradient(self, out_grad, node):
        raise NotImplementedError("Gradient for EWiseEq not implemented yet.")

class EWiseEq(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a == b

    def gradient(self, out_grad, node):
        raise NotImplementedError("Gradient for EWiseEq not implemented yet.")

class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar, dtype=a.dtype)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (out_grad * self.scalar * a**(self.scalar - 1),)

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.divide(a, b, dtype=a.dtype)

    def gradient(self, out_grad, node):
        a, b = node.inputs
        da = out_grad * b**-1
        db = out_grad * -a/(b**2)

        assert da.shape == a.shape, f"a: {a.shape}, da: {da.shape}"
        assert db.shape == b.shape, f"b: {b.shape}, db: {db.shape}"

        return da, db


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar, dtype=a.dtype)

    def gradient(self, out_grad, node):
        return (out_grad * (1/self.scalar),)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes:
            axis1, axis2 = self.axes
        else:
            axis1, axis2 = -1, -2
        return array_api.swapaxes(a, axis1, axis2)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad.reshape(a.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        """The reverse would add up all the gradient associated with the broadcasted gradient."""
        a = node.inputs[0]
        da = sum_to_shape(out_grad, a.shape)
        assert a.shape == da.shape, f"Expect after: {a.shape} === {da.shape}"
        return da

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


def size(shape):
    if len(shape) == 0:
        return 0
    return functools.reduce(lambda a,b: a*b, shape, 1)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        """
        Because Summation would remove a dimension, here we added back the removed dimension, 
        and let broadcast takes care of the gradient computation for each element's contribution.
        """
        a = node.inputs[0]

        # support None, rebroad cast to [1, ..., 1]
        if self.axes is None:
            axes = []
        elif not isinstance(self.axes, (list, tuple)):
            axes = [self.axes]
        else:
            axes = self.axes
        
        # support negative index of axes
        axes = [a if a >= 0 else len(a.shape) - a for a in axes]

        # NOTE: convert back to the original shape before broadcasting
        origin_shape = [1 if i in axes else a.shape[i]
                        for i, _ in enumerate(a.shape)]

        # print(f"[Summation] outgrad: {out_grad.shape} -> origin {origin_shape} -> broadcast {a.shape}")

        if size(origin_shape) == size(out_grad.shape):
            out_grad = out_grad.reshape(origin_shape)

        da = out_grad.broadcast_to(a.shape)
        assert a.shape == da.shape, f"Expect after: {a.shape} === {da.shape}"
        return da

def summation(a, axes=None):
    return Summation(axes)(a)


def sum_to_shape(x, to_shape):
    axes_to_sum = []
    # NOTE: right aligned
    for i in range(len(x.shape)):
        bi = -1 - i
        if i >= len(to_shape):
            axes_to_sum.append(bi)
        elif x.shape[bi] != to_shape[bi]:
            axes_to_sum.append(bi)
    # print(f"Before: {x.shape} -> {to_shape}")
    # print(f"axes_to_sum: ", axes_to_sum)

    if axes_to_sum:
        x = x.sum(axes=tuple(axes_to_sum))

    # TODO: Is this always safe for multiple ones????
    # sum [n, 1, n, 1] -> [n, n]
    # reshape: [n, n] -> [n, 1, n, 1]
    x = x.reshape(to_shape)

    assert x.shape == to_shape, f"Expect after: {x.shape} === {to_shape}"

    return x

class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        a, b = node.inputs

        # print(f"\n\na: {a.shape}, a.T: {a.transpose().shape} o: {out_grad.shape}, b:{b.shape}, b.T: {b.transpose().shape}")

        da = out_grad @ b.transpose()  # transpose is by default last 2 axises
        db = a.transpose() @ out_grad

        da = sum_to_shape(da, a.shape)
        db = sum_to_shape(db, b.shape)

        assert da.shape == a.shape, f"a: {a.shape}, da: {da.shape}"
        assert db.shape == b.shape, f"b: {b.shape}, da: {db.shape}"
        return da, db


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return (-out_grad,)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (out_grad * a**-1,)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (out_grad * array_api.exp(a),)

def exp(a):
    return Exp()(a)

class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        # NOTE: Exception to look backward at input
        mask = a.realize_cached_data() > 0
        return (out_grad * mask,)

def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def restore_shape(self, Z: Tensor):
        # Reshape the maxZ to match Z shape
        # The dimension which is summed out would be 1, rest unchanged.
        new_shape = [1] * len(Z.shape)
        for i in range(len(new_shape)):
            if self.axes and i not in self.axes:
                new_shape[i] = Z.shape[i]
        return tuple(new_shape)


    def compute(self, Z):
        maxZ = array_api.max(Z, axis=self.axes)
        new_shape = self.restore_shape(Z)
        return array_api.log(
            array_api.sum(
                array_api.exp(Z - maxZ.reshape(new_shape)),
                axis=self.axes)) + maxZ

    def gradient(self, out_grad, node):
        """
        f(a) = a - max(a)
        g(x) = log(sum(exp(x)))

        df/da = [1, 1, ... 0, ... 1]
        dg/dx = exp(x) / sum(exp(x))

        Q: why the df/da is not used here?
        """
        input = node.inputs[0].realize_cached_data()

        max_in = array_api.max(input, axis=self.axes)
        input_exp = array_api.exp(input - max_in.reshape(self.restore_shape(input)))

        input_exp_sum = array_api.sum(input_exp, axis=self.axes)
        gradient = input_exp / input_exp_sum.reshape(self.restore_shape(input))

        # NOTE: out_grad also has dimension reduced, here restore it
        # for broadcasting to work properly
        return out_grad.reshape(self.restore_shape(input)) * needle.Tensor(gradient, dtype=out_grad.dtype)

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
