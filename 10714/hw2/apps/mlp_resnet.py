import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

# parameter count: 3*dim + 3*hidden_dim + 2*dim*hidden_dim
def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    net = nn.Sequential(
        nn.Linear(dim, hidden_dim),  # (dim + 1) * hidden_dim
        norm(hidden_dim),           # 2 * hidden_dim
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_dim, dim),  # (hidden_dim + 1) * dim
        norm(dim),                  # 2 * dim
    )
    return nn.Sequential(
        nn.Residual(net),
        nn.ReLU()
    )

def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    modules = ([
        nn.Flatten(), # first flat all the 3-d image to 1-d data
        nn.Linear(dim, hidden_dim),
        nn.ReLU()] +
        [ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for i in range(num_blocks)] +
        [nn.Linear(hidden_dim, num_classes)])
    resnet = nn.Sequential(*modules)
    return resnet

def error_count(logits, Yb) -> float: 
        logits_data = logits.detach()
        Xprob = (logits_data.exp() / logits_data.exp().sum(axes=(1,)).reshape((-1, 1)))
        Xpred = Xprob.argmax(axis=1)
        error_c = (Xpred != Yb).sum().numpy()
        return error_c

def epoch(dataloader, model, opt=None):
    """
    In each epoch, the model would:
    - Run the forward path and gets out the logits
    - Use the logits to compute the accurancy and loss
    - Use an optimizer to run backward path and update all the parameters
    """
    np.random.seed(4)
    if opt is not None:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_error_count = 0
    running_N = 0
    for batch in dataloader:
        Xb, Yb = batch
        batch_size = Xb.shape[0]
        # forward
        logits = model(Xb)
        loss = nn.SoftmaxLoss()(logits, Yb)  # TODO: Use a new loss head for every epoch?

        # book keeping, no_grad
        running_error_count += error_count(logits, Yb)
        running_loss += loss.numpy() * batch_size
        running_N += batch_size 

        # backward pass, takes place on every batch. "tiny little steps"
        if model.training:
            opt.reset_grad()
            loss.backward()
            opt.step()

    return running_error_count / running_N, running_loss / running_N


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    """
    Train assembles all components together
    - Data pipeline and data loader
    - Hypothesis model
    - Optimizer
    Setup epoch and monitor the training.
    """
    np.random.seed(4)
    train_dataset = ndl.data.MNISTDataset(\
            f"./{data_dir}/train-images-idx3-ubyte.gz",
            f"./{data_dir}/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             shuffle=True)

    test_dataset = ndl.data.MNISTDataset(\
            f"./{data_dir}/t10k-images-idx3-ubyte.gz",
            f"./{data_dir}/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
        test_error, test_loss = epoch(test_dataloader, model)
        if i % 1 == 0:
            print(f"Epoch {i}: train error: {train_error}, test error: {test_error}")
            print(f"Epoch {i}: train loss: {train_loss}, test loss: {test_loss}")
        
    return train_error, train_loss, test_error, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
