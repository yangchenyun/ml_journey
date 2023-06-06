# RNN implementation from scratch

# Goal 1: Implement both forward and backward path of RNN
# - Setup test cases with torch implementation
# - Pass forward test
# - Pass backwrd test

# Goal 2: Implement LSTM
# - forward
# - backward

# Goal 3: Implement encoder / decoder
# - Download dataset
# - k-hot encoder / decoder
# - train a few loop
# - sampling from the model

# %%
import torch
import numpy as np

input_size = 10
output_size = 10
hidden_size = 100
seq_len = 10

# %% Reference implementation with pytorch
class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn_layer = torch.nn.RNNCell(input_size, hidden_size, bias=True)
        self.linear_layer = torch.nn.Linear(hidden_size, output_size)

        # NOTE: used for reference
        self.model = torch.nn.Sequential(self.rnn_layer, self.linear_layer)

    def forward(self, in_seq, h_0):
        """
        in_seq: A sequence in shape (N, In)
        """
        t = len(in_seq)
        h_i = h_0
        out = []
        for i in range(t):
            h_i = self.rnn_layer(in_seq[i], h_i)
            out_i = self.linear_layer(h_i)
            out.append(out_i)
        return torch.stack(out, 0)

model = RNNModel(input_size, hidden_size, output_size)
input_seq = torch.randn(seq_len, input_size)
h_0 = torch.randn(hidden_size)
output = model(input_seq, h_0)
print(output.shape)

# %% Implementation from scratch
def init_param(shape, k):
    return np.random.uniform(-1, 1, size=shape) * np.sqrt(k)

class RNNBasic:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        k = 1/hidden_size
        self.W_hh = init_param((hidden_size, hidden_size), k)
        self.W_hi = init_param((hidden_size, input_size), k)
        self.B_h = init_param((hidden_size,), 0)
        self.B_i = init_param((hidden_size,), 0)
        self.W_yh = init_param((output_size, hidden_size), k)  # TODO: use Xavier normalization here
        self.B_y = init_param((output_size,), 0)
        self.h_i_cache = None

    def forward(self, in_seq, h_0):
        # Index starts at 1
        out_seq = np.empty((len(in_seq) + 1, self.output_size))
        self.h_i_cache = np.empty((len(in_seq) + 1, self.hidden_size))
        h_i = h_0
        self.h_i_cache[0] = h_i

        for i in range(1, len(in_seq) + 1):  # NOTE: Only in_seq is zero-based
            h_i = np.tanh(self.W_hi @ in_seq[i - 1] + self.B_i + self.W_hh @ h_i + self.B_h)
            out_seq[i] = self.W_yh @ h_i + self.B_y
            self.h_i_cache[i] = h_i

        self.in_seq_cache = in_seq
        
        return out_seq[1:]

    def backward(self, dout_seq):
        """Computes the gradient for each parameters.
        dout_seq has shape (Len_Seq, output_size)
        """
        dW_hh = np.zeros_like(self.W_hh)
        dW_hi = np.zeros_like(self.W_hi)
        dB_h = np.zeros_like(self.B_h)
        dB_i = np.zeros_like(self.B_i)
        dW_yh = np.zeros_like(self.W_yh)
        dB_y = np.zeros_like(self.B_y)
        
        dh_i_cache = np.empty((len(dout_seq) + 1, self.hidden_size))

        for i in reversed(range(1, len(dout_seq) + 1)):  # NOTE: Only dout_seq is zero-based
            # linear layer back propagation
            dW_yh += np.outer(dout_seq[i - 1], self.h_i_cache[i].T)
            dB_y += dout_seq[i - 1]
            dh_i_cache[i] = dout_seq[i - 1] @ self.W_yh

            # d(tanh(x))/dx = 1 - (tanh(x))^2.
            dh_i_linear = dh_i_cache[i] * (1 - np.power(self.h_i_cache[i], 2))
            dW_hh += np.outer(dh_i_linear, self.h_i_cache[i - 1].T)
            dW_hi += np.outer(dh_i_linear, self.in_seq_cache[i - 1].T) # NOTE: in_seq is 0-indexed
            dB_h += dh_i_linear
            dB_i += dh_i_linear
            dh_i_cache[i - 1] = dh_i_linear @ self.W_hh

        # Check if the shapes of gradients all match with parameters
        assert dW_hh.shape == self.W_hh.shape
        assert dW_hi.shape == self.W_hi.shape
        assert dB_h.shape == self.B_h.shape
        assert dB_i.shape == self.B_i.shape
        assert dW_yh.shape == self.W_yh.shape
        assert dB_y.shape == self.B_y.shape

        return (dW_hh, dW_hi, dB_h, dB_i, dW_yh, dB_y)


# %% Forward comparison
model = RNNModel(input_size, hidden_size, output_size)
m2 = RNNBasic(input_size, hidden_size, output_size)

for name, param in model.named_parameters():
    if param.requires_grad:
        if name == 'rnn_layer.weight_ih':
            param.data = torch.Tensor(m2.W_hi)
        elif name == 'rnn_layer.weight_hh':
            param.data = torch.Tensor(m2.W_hh)
        elif name == 'rnn_layer.bias_ih':
            param.data = torch.Tensor(m2.B_i)
        elif name == 'rnn_layer.bias_hh':
            param.data = torch.Tensor(m2.B_h)
        elif name == 'linear_layer.weight':
            param.data = torch.Tensor(m2.W_yh)
        elif name == 'linear_layer.bias':
            param.data = torch.Tensor(m2.B_y)
        else:
            print("Unexpected model parameter", name, param)

input_seq = torch.randn(seq_len, input_size)
hidden_0 = torch.zeros(hidden_size)
output1 = model(input_seq, hidden_0)
output2 = m2.forward(input_seq.numpy(), hidden_0.numpy())

assert output1.shape == output2.shape
np.allclose(output1.detach().numpy(), output2, rtol=1e-5, atol=1e-5)

# %% Backward comparison
output1 = model(input_seq, hidden_0)
output2 = m2.forward(input_seq.numpy(), hidden_0.numpy())
model.zero_grad()
loss = torch.sum(output1 - input_seq)  # NOTE: For simple dout.
loss.backward()

dout_seq = np.ones_like(output2)
(dW_hh, dW_hi, dB_h, dB_i, dW_yh, dB_y) = m2.backward(dout_seq)

def gradient_comp(grad1, grad2):
    gradient_diff = np.abs(grad1 - grad2)
    max_diff = gradient_diff.max()
    return max_diff.numpy()

for name, param in model.named_parameters():
    if param.requires_grad:
        if name == 'rnn_layer.weight_ih':
            assert param.shape == dW_hi.shape
            print("max diff of dW_hi:", gradient_comp(param.grad, dW_hi))
            # assert np.allclose(param.grad.numpy(), dW_hi, rtol=1e-3, atol=1e-3)
        elif name == 'rnn_layer.weight_hh':
            assert param.shape == dW_hh.shape
            print("max diff of dW_hh:", gradient_comp(param.grad, dW_hh))
            # assert np.allclose(param.grad.numpy(), dW_hh, rtol=1e-3, atol=1e-3)
        elif name == 'rnn_layer.bias_ih':
            assert param.shape == dB_i.shape
            print("max diff of dB_i:", gradient_comp(param.grad, dB_i))
            # assert np.allclose(param.grad.numpy(), dB_i, rtol=1e-3, atol=1e-3)
        elif name == 'rnn_layer.bias_hh':
            assert param.shape == dB_h.shape
            print("max diff of dB_h:", gradient_comp(param.grad, dB_h))
            # assert np.allclose(param.grad.numpy(), dB_h, rtol=1e-2, atol=1e-2)
        elif name == 'linear_layer.weight':
            assert param.shape == dW_yh.shape
            assert np.allclose(param.grad.numpy(), dW_yh, rtol=1e-3, atol=1e-3)
        elif name == 'linear_layer.bias':
            assert param.shape == dB_y.shape
            assert np.allclose(param.grad.numpy(), dB_y, rtol=1e-3, atol=1e-3)
        else:
            print("Unexpected model parameter", name, param)


# %% Reference implementation with pytorch
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size, bias=True)
        self.linear_layer = torch.nn.Linear(hidden_size, output_size)

        # NOTE: used for reference
        self.model = torch.nn.Sequential(self.lstm_cell, self.linear_layer)

    def forward(self, in_seq, init_0):
        """
        in_seq: A sequence in shape (N, In)
        """
        h_i, c_i = init_0
        out = []
        for i in range(len(in_seq)):
            h_i, c_i = self.lstm_cell(in_seq[i], (h_i, c_i))
            out_i = self.linear_layer(h_i)
            out.append(out_i)
        return torch.stack(out, 0)

model = LSTMModel(input_size, hidden_size, output_size)
input_seq = torch.randn(seq_len, input_size)
h_0 = torch.randn(hidden_size)
c_0 = torch.randn(hidden_size)
output = model(input_seq, (h_0, c_0))
print(output.shape)

# %% Implementation from scratch
class LSTMBasic:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        k = 1/hidden_size
        # NOTE: stack the weights together
        self.W_hh = init_param((4*hidden_size, hidden_size), k)
        self.W_hi = init_param((4*hidden_size, input_size), k)
        self.B_h = init_param((4*hidden_size,), 0)
        self.B_i = init_param((4*hidden_size,), 0)
        self.W_yh = init_param((output_size, hidden_size), k)  # TODO: use Xavier normalization here
        self.B_y = init_param((output_size,), 0)

    def forward(self, in_seq, init_0):
        # Index starts at 1
        out_seq = np.empty((len(in_seq) + 1, self.output_size))
        h_i, c_i = init_0

        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        for i in range(1, len(in_seq) + 1):  # NOTE: Only in_seq is zero-based
            h_linear = self.W_hi @ in_seq[i - 1] + self.B_i + self.W_hh @ h_i + self.B_h

            i_ = sigmoid(h_linear[:self.hidden_size])
            f = sigmoid(h_linear[self.hidden_size:2*self.hidden_size])
            g = np.tanh(h_linear[2*self.hidden_size:3*self.hidden_size])
            o = sigmoid(h_linear[3*self.hidden_size:])

            c_i = f * c_i + i_ * g
            h_i = o * np.tanh(c_i)

            out_seq[i] = self.W_yh @ h_i + self.B_y
        
        return out_seq[1:]

model = LSTMModel(input_size, hidden_size, output_size)
m2 = LSTMBasic(input_size, hidden_size, output_size)

for name, param in model.named_parameters():
    if param.requires_grad:
        if name == 'lstm_cell.weight_ih':
            param.data = torch.Tensor(m2.W_hi)
        elif name == 'lstm_cell.weight_hh':
            param.data = torch.Tensor(m2.W_hh)
        elif name == 'lstm_cell.bias_ih':
            param.data = torch.Tensor(m2.B_i)
        elif name == 'lstm_cell.bias_hh':
            param.data = torch.Tensor(m2.B_h)
        elif name == 'linear_layer.weight':
            param.data = torch.Tensor(m2.W_yh)
        elif name == 'linear_layer.bias':
            param.data = torch.Tensor(m2.B_y)
        else:
            print("Unexpected model parameter", name, param)

input_seq = torch.randn(seq_len, input_size)
h_0 = torch.zeros(hidden_size)
c_0 = torch.zeros(hidden_size)

output1 = model(input_seq, (h_0, c_0))
output2 = m2.forward(input_seq.numpy(), (h_0.numpy(), c_0.numpy()))

assert output1.shape == output2.shape
np.allclose(output1.detach().numpy(), output2, rtol=1e-5, atol=1e-5)
# %%
