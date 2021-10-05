import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import gzip
import time
import struct
import os

np.random.seed(0)
    
def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm, drop_prob=0.1):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm, drop_prob=0.1):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_epoch(dataloader, model, loss_fn, opt):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def evaluate(dataloader, model, loss_fn=nn.SoftmaxLoss()):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def num_params(model):
    return np.sum([np.prod(x.shape) for x in model.parameters()])

if __name__ == "__main__":
    train_mnist(data_dir="../data")