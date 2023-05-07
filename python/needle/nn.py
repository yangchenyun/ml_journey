"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from functools import reduce


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features, requires_grad=True, device=device, dtype=dtype))
        # NOTE: usually bias could be initialized as zero
        self.bias = Parameter(init.kaiming_uniform(self.out_features, 1, requires_grad=True, device=device, dtype=dtype).transpose() if bias else None)

    def forward(self, X: Tensor) -> Tensor:
        out = X @ self.weight
        # NOTE: Apply bias for every output vector
        out = out + self.bias.broadcast_to(out.shape) if self.bias else out
        return out

class Flatten(Module):
    def forward(self, X):
        return X.reshape((X.shape[0], -1))

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        return reduce(lambda acc, m: m.forward(acc), self.modules, x)

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        b = logits.shape[0]
        axis = len(logits.shape) - 1 # last dimension as data
        n = logits.shape[axis]
        H_y = logits * init.one_hot(n, y)  # pluck out the encoding label == y
        delta = logits.logsumexp(axes=(axis,)) - H_y.sum(axes=(axis,))
        return delta.sum() / b

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        n = x.shape[1]

        if self.training:
            # NOTE: Calculate stats over the batch dimension now
            batch_mean = (x.sum(axes=(0,)) / x.shape[0])
            batch_var = (((x - batch_mean.broadcast_to(x.shape)) ** 2).sum(axes=(0,)) / x.shape[0])  # biased variance
            self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean.detach() * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + batch_var.detach() * self.momentum

            batch_mean = batch_mean.broadcast_to(x.shape)

            batch_var = batch_var.broadcast_to(x.shape)
            normalized = (x - batch_mean) / (batch_var + self.eps)**0.5
        else:
            normalized = (x - self.running_mean) / (self.running_var + self.eps)**0.5
            
        broadcasted_result = normalized * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        result = normalized * self.weight + self.bias

        assert np.all(broadcasted_result.numpy() == result.numpy()), "Broadcasting should not change the result"

        # TODO: Why not use weights.broadcast_to would make numeric difference in Adam?
        # Because gradient propagation for self.weight and self.bias?
        # import pdb; pdb.set_trace()
        # return result
        return broadcasted_result

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        n = x.shape[1]

        # NOTE: Calculate stats over the feature dimension
        # NOTE: Would fail if not broadcasting, the grad shape won't agree
        layer_mean = (x.sum(axes=(1,)) / n).reshape((b, 1)).broadcast_to(x.shape)
        layer_var = (((x - layer_mean) ** 2).sum(axes=(1,)) / n).reshape((b, 1)).broadcast_to(x.shape)
        normalized = (x - layer_mean) / (layer_var + self.eps)**0.5

        broadcasted_result = normalized * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        result = normalized * self.weight + self.bias

        assert np.all(broadcasted_result.numpy() == result.numpy()), "Broadcasting should not change the result"
        return broadcasted_result

class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            drop = init.randb(*x.shape, p=self.p)
            return drop * x / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)