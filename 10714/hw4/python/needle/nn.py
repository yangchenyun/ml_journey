"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from functools import reduce
import pickle
import math


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

    def __getstate__(self):
        import types
        state = self.__dict__.copy()
        for key, value in state.items():
            if isinstance(value, Module) or isinstance(value, Parameter):
                state[key] = pickle.dumps(value)
            elif isinstance(value, (list, tuple)):
                state[key] = type(value)(pickle.dumps(v) if isinstance(v, (Module, Parameter)) else v for v in value)
            elif isinstance(value, dict):
                state[key] = {k: pickle.dumps(v) if isinstance(v, (Module, Parameter)) else v for k, v in value.items()}
            elif isinstance(value, types.ModuleType):
                del state[key]  # Don't try to pickle modules
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            if isinstance(value, (bytes)):
                state[key] = pickle.loads(value)
            elif isinstance(value, (list, tuple)):
                state[key] = type(value)(pickle.loads(v) if isinstance(v, (bytes)) else v for v in value)
            elif isinstance(value, dict):
                state[key] = {k: pickle.loads(v) if isinstance(v, (bytes)) else v for k, v in value.items()}
        self.__dict__.update(state)


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

class LeakyReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.leaky_relu(x)

class ELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.elu(x)

class SeLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.selu(x)

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


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
        H_y = logits * init.one_hot(n, y, device=y.device)  # pluck out the encoding label == y
        delta = logits.logsumexp(axes=(axis,)) - H_y.sum(axes=(axis,))
        return delta.sum() / b

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
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
            running_mean = self.running_mean.broadcast_to(x.shape)
            running_var = self.running_var.broadcast_to(x.shape)
            normalized = (x - running_mean) / (running_var + self.eps)**0.5
            
        broadcasted_result = normalized * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        result = normalized * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)

        np.testing.assert_allclose(broadcasted_result.numpy(), result.numpy(), atol=1e-5, rtol=1e-5, 
                                   err_msg="Broadcasting should not change the result")

        # TODO: Why not use weights.broadcast_to would make numeric difference in Adam?
        # Because gradient propagation for self.weight and self.bias?
        # import pdb; pdb.set_trace()
        # return result
        return broadcasted_result


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


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
            # NOTE: another tricky implementation details
            # The probability here is the probability of keeping the value
            keep = init.randb(*x.shape, p=1-self.p)
            return keep * x / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(kernel_size*kernel_size*in_channels,   # fan_in
                                 kernel_size*kernel_size*out_channels,  # fan_out
                                 shape=(kernel_size, kernel_size, in_channels, out_channels), device=device, dtype=dtype)
        )
        self.bias = None
        if bias:
            self.bias = Parameter(
                init.rand(out_channels, low=-1, high=1, device=device, dtype=dtype) / (in_channels * kernel_size**2)**0.5)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N,C,H,W = x.shape

        padding = ((H - 1)*1 - H + self.kernel_size)//2

        x = x.transpose((1, 2)).transpose((2, 3))
        # NOTE: result in (N, H, W, C_out)
        result = ops.conv(x, self.weight, self.stride, padding)
        if self.bias:
            result += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(result.shape)
        result = result.transpose((2, 3)).transpose((1, 2))
        return result
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        ### BEGIN YOUR SOLUTION
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        k = math.sqrt(1/hidden_size)

        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-k, high=k, device=device, requires_grad=True))
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-k, high=k, device=device, requires_grad=True))

        if self.bias:
            self.bias_hh = Parameter(init.rand(1, hidden_size, low=-k, high=k, device=device, requires_grad=True))
            self.bias_ih = Parameter(init.rand(1, hidden_size, low=-k, high=k, device=device, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device)

        h_prime = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            h_prime += self.bias_ih.broadcast_to(h_prime.shape) + self.bias_hh.broadcast_to(h_prime.shape)

        if self.nonlinearity == 'tanh':
            h_prime = h_prime.tanh()
        elif self.nonlinearity == 'relu':
            h_prime = h_prime.relu()


        return h_prime
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)] + [
                RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        T,B,_ = X.shape
        H = init.zeros(T, self.num_layers, B, self.hidden_size, device=X.device)

        # NOTE: Because lack of support of indexing, convert to python array.

        if h0 is None:
            h0 = [init.zeros(B, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
        else:
            h0 = [t.reshape(t.shape[1:]) for t in ops.split(h0, 0)]

        input_outputs = [t.reshape(t.shape[1:]) for t in ops.split(X, 0)]
        h_n = []
        for l in range(self.num_layers):
            h = h0[l]
            for t in range(0, T):
                h = self.rnn_cells[l](input_outputs[t], h)
                input_outputs[t] = h  # NOTE: Discard input at time t
            h_n.append(h)

        return ops.stack(input_outputs, 0), ops.stack(h_n, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        k = math.sqrt(1/hidden_size)

        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-k, high=k, device=device), requires_grad=True)
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-k, high=k, device=device), requires_grad=True)

        if self.bias:
            self.bias_hh = Parameter(init.rand(1, 4*hidden_size, low=-k, high=k, device=device), requires_grad=True)
            self.bias_ih = Parameter(init.rand(1, 4*hidden_size, low=-k, high=k, device=device), requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h0 = init.zeros(X.shape[0], self.hidden_size, device=self.device)
            c0 = init.zeros(X.shape[0], self.hidden_size, device=self.device)
        else:
            h0, c0 = h

        # (B, hidden_size*4)
        h_linear = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            h_linear += self.bias_ih.broadcast_to(h_linear.shape) + self.bias_hh.broadcast_to(h_linear.shape)

        def split_list(lst, k):
            n = len(lst)
            return [lst[i * n // k: (i + 1) * n // k] for i in range(k)]

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        i, f, g, o = split_list([p.reshape((p.shape[0],)) for p in ops.split(h_linear, axis=1)], 4)
        i = ops.stack(i, axis=1).sigmoid()
        f = ops.stack(f, axis=1).sigmoid()
        g = ops.stack(g, axis=1).tanh()
        o = ops.stack(o, axis=1).sigmoid()

        c_prime = c0 * f + i * g
        h_prime = o * c_prime.tanh()

        return h_prime, c_prime
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)] + [
                LSTMCell(hidden_size, hidden_size, bias, device, dtype) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        T,B,_ = X.shape
        H = init.zeros(T, self.num_layers, B, self.hidden_size, device=X.device)
        C = init.zeros(T, self.num_layers, B, self.hidden_size, device=X.device)

        if h is None:
            h0 = [init.zeros(B, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
            c0 = [init.zeros(B, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
        else:
            h0 = [t.reshape(t.shape[1:]) for t in ops.split(h[0], 0)]
            c0 = [t.reshape(t.shape[1:]) for t in ops.split(h[1], 0)]

        input_outputs = [t.reshape(t.shape[1:]) for t in ops.split(X, 0)]
        h_n = []
        c_n = []
        for l in range(self.num_layers):
            h = h0[l]
            c = c0[l]
            for t in range(0, T):
                h, c = self.lstm_cells[l](input_outputs[t], (h, c))
                input_outputs[t] = h  # NOTE: Discard input at time t
            h_n.append(h)
            c_n.append(c)

        return ops.stack(input_outputs, 0), (ops.stack(h_n, 0), ops.stack(c_n, 0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        T, B = x.shape
        # NOTE: x is detached from the computational graph
        x_data = x.detach().numpy().astype(np.int32).reshape(-1)
        one_hot_encoded = np.eye(self.num_embeddings)[x_data]
        one_hot_encoded = Tensor(one_hot_encoded, device=self.device, dtype=self.dtype)
        output = one_hot_encoded @ self.weight
        output = output.reshape((T, B, -1))
        return output
        ### END YOUR SOLUTION
