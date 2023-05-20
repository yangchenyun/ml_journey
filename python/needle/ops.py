"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy as np
import functools
import needle
from needle import init

# import numpy as array_api
from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


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
                in_grad.append(init.zeros_like(value))
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
        return a + np.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


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
        return a * np.float32(self.scalar)

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
        return a / b

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
        return a / np.float32(self.scalar)

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
        return a.compact().reshape(self.shape)

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
        assert len(a.shape) == len(self.shape), "BroadcastTo only support same dimension broadcasting."
        axes = tuple(i for i, (x, y) in enumerate(zip(a.shape, self.shape)) if x != y)
        da = summation(out_grad, axes, keepdims=True)
        assert a.shape == da.shape, f"Expect after: {a.shape} === {da.shape}"
        return da

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


def size(shape):
    if len(shape) == 0:
        return 0
    return functools.reduce(lambda a,b: a*b, shape, 1)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims: bool = False):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        return array_api.summation(a, axis=self.axes, keepdims=self.keepdims)

    def gradient(self, out_grad, node):
        """
        Because Summation would remove a dimension, here we added back the removed dimension, 
        and let broadcast takes care of the gradient computation for each element's contribution.
        """
        a = node.inputs[0]

        # support None, rebroad cast to [1, ..., 1]
        if self.axes is None:
            axes = [i for i in range(len(a.shape))]
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

def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs

        # print(f"\n\na: {a.shape}, a.T: {a.transpose().shape} o: {out_grad.shape}, b:{b.shape}, b.T: {b.transpose().shape}")

        da = out_grad @ b.transpose()  # transpose is by default last 2 axises
        db = a.transpose() @ out_grad


        a_axes = tuple(i for i, (x, y) in enumerate(zip(a.shape, da.shape)) if x != y)
        b_axes = tuple(i for i, (x, y) in enumerate(zip(b.shape, db.shape)) if x != y)
        da = summation(da, a_axes, keepdims=True)
        db = summation(db, a_axes, keepdims=True)

        assert da.shape == a.shape, f"a: {a.shape}, da: {da.shape}"
        assert db.shape == b.shape, f"b: {b.shape}, da: {db.shape}"
        return da, db


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

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
            if self.axes is not None:
                if isinstance(self.axes, tuple):
                    if i not in self.axes: new_shape[i] = Z.shape[i]
                elif isinstance(self.axes, int):
                    if i != self.axes: new_shape[i] = Z.shape[i]
        return tuple(new_shape)


    def compute(self, Z):
        maxZ = array_api.max(Z, axis=self.axes)
        new_shape = self.restore_shape(Z)
        return ((Z - maxZ.compact().reshape(new_shape).broadcast_to(Z.shape))
                .exp()
                .sum(axis=self.axes)
                .log()) + maxZ

    def gradient(self, out_grad, node):
        """
        f(a) = a - max(a)
        g(x) = log(sum(exp(x)))

        df/da = [1, 1, ... 0, ... 1]
        dg/dx = exp(x) / sum(exp(x))

        Q: why the df/da is not used here?
        """
        input = node.inputs[0].realize_cached_data()

        max_in = array_api.max(input, axis=self.axes, keepdims=True)
        input_exp = array_api.exp(input - max_in.broadcast_to(input.shape))

        input_exp_sum = array_api.summation(input_exp, axis=self.axes, keepdims=True)
        gradient = input_exp / input_exp_sum.broadcast_to(input.shape)

        # NOTE: out_grad also has dimension reduced, here restore it
        # for broadcasting to work properly
        return out_grad.reshape(self.restore_shape(input)).broadcast_to(gradient.shape) * needle.Tensor(gradient)

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def __init__(self):
        self.cached_tanh = None

    def compute(self, a):
        a_exp2 = a.exp() * a.exp()
        return (a_exp2 - 1) / (a_exp2 + 1)

    def gradient(self, out_grad, node):
        input = node.inputs[0]
        tanh = self.compute(input)
        return out_grad * (1 - tanh ** 2)

def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        return array_api.stack(*args, axis=self.axis)

    def gradient(self, out_grad, node):
        input_arrays = node.inputs[0]  # input has been made as a tuple
        out = split(out_grad, self.axis)
        assert len(out) == len(input_arrays), "Gradient length mismatch"
        return make_tuple(*[out[i].reshape(in_ary.shape) 
                           for i, in_ary in enumerate(input_arrays)])

def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        return array_api.split(A, A.shape[self.axis], axis=self.axis)

    def gradient(self, out_grad, node):
        In = node.inputs[0]
        out = stack(out_grad, axis=self.axis).reshape(In.shape)
        assert len(out.shape) == len(In.shape)
        return out


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.dilate(a, self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.undilate(a, self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride or 1
        self.padding = padding or 0

    def compute(self, A, B):
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        P = self.padding
        S = self.stride

        A_pad = A.pad(axes=((0, 0), (P, P), (P, P), (0, 0)))

        Ns, Hs, Ws, Cs = A_pad.strides
        conv_strides = (Ns, Hs*S, Ws*S, Hs, Ws, Cs)
        conv_shape = tuple(np.array([N, (H+2*P-K)/S + 1, (W+2*P-K)/S + 1], dtype=np.int64))

        inner_dim = K * K * C_in
        out = A_pad.as_strided(conv_shape + (K, K, C_in), conv_strides).compact()

        # Flatten the inner dimensions
        out = out.reshape((out.size//inner_dim, inner_dim)) @ B.compact().reshape((inner_dim, C_out))
        out = out.reshape(conv_shape + (C_out,))
        return out

    def gradient(self, out_grad, node):
        Z, W = node.inputs

        N,Hz,Wz,C_in = Z.shape
        _,Ho,Wo,_ = out_grad.shape
        K,_,_,C_out = W.shape

        revP = K-1-self.padding
        if self.stride > 1:
            out_grad = dilate(out_grad, (1,2), self.stride - 1)
            # TODO: Reverse calcuate the expected dimensions
            H_g = (Hz - 1) + K - 2 * revP
            W_g = (Wz - 1) + K - 2 * revP
            assert H_g == out_grad.shape[1]
            assert W_g == out_grad.shape[2]
            # slice operator missing

        # flip kernel dimensions
        # swap C_in and C_out
        # out_grad: N,H,W,C_out
        # dW: K,K,C_in,C_out -> K,K,C_out,C_in
        fW = flip(W, (0, 1))
        dZ = conv(out_grad, transpose(fW, (2, 3)), padding=revP)
        assert dZ.shape == Z.shape

        # Z: N,H,W,C_in -> C_in,H,W,N, treating N as input channels
        # out_grad: N,H,W,C_out -> W,H,N,C_out -> H,W,N,C_out, treating N as input channels, H,W as kernel window
        # dW: C_in,K,K,C_out -> K,K,C_in,C_out (keep the order of two kernel dimensions)
        tZ = transpose(Z, (0, 3))
        tOut_grad = transpose(transpose(out_grad, (0, 2)), (0, 1))
        tW = conv(tZ, tOut_grad, padding=self.padding) # apply the same padding as in forward pass
        dW = transpose(transpose(tW, (0, 2)), (0, 1)) 

        assert dW.shape == W.shape
        return dZ, dW


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)