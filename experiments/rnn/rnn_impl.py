# RNN implementation from scratch

# Goal 1: Implement both forward and backward path of RNN
# - Setup test cases with torch implementation
# - Pass forward test
# - Pass backwrd test

# Goal 2: Implement encoder / decoder
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
seq_len = 1

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
        hidden = self.rnn_layer(in_seq, h_0)
        out = self.linear_layer(hidden)
        return out

# model = RNNModel(input_size, hidden_size, output_size)
# input_seq = torch.randn(seq_len, input_size)
# h_0 = torch.randn(1, hidden_size)
# output = model(input_seq, h_0)

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

    def forward(self, in_seq, h_0):
        # Index starts at 1
        out_seq = np.empty((len(in_seq) + 1, self.output_size))
        h_i = h_0[0, :] # TODO: assuming single batch here

        for i in range(1, len(in_seq) + 1):  # NOTE: 
            h_i = np.tanh(self.W_hi @ in_seq[i - 1] + self.B_i + self.W_hh @ h_i + self.B_h)
            print(h_i.shape)
            out_seq[i] = self.W_yh @ h_i + self.B_y
        
        return out_seq[1:]

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
hidden_0 = torch.zeros((1, hidden_size))
output1 = model(input_seq, hidden_0)
output2 = m2.forward(input_seq.numpy(), hidden_0.numpy())

assert output1.shape == output2.shape
np.allclose(output1.detach().numpy(), output2, rtol=1e-5, atol=1e-5)

# %% Backward comparison
# loss = torch.mean((output - input_batch) ** 2)
# loss.backward()
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.grad.shape)
