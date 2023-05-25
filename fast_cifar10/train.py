#%%
import torch
import torch.nn as nn
import numpy as np
from functools import partial

import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR

import wandb

import sys
import logging

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=log_format)
logger = logging.getLogger()
#%%
class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

class Add(nn.Module):
    def __init__(self, fn1: nn.Module, fn2: nn.Module):
        super().__init__()
        self.fn1 = fn1
        self.fn2 = fn2

    def forward(self, x):
        return self.fn1(x) + self.fn2(x)

def prep_block(c_in, c_out, **kw):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw),
        nn.BatchNorm2d(num_features=c_out, **kw),
        nn.ReLU(),
    )

def res_block(c_in, c_out, stride, **kw):
    branch = nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False, **kw),
        nn.BatchNorm2d(c_out, **kw),
        nn.ReLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw),
    )
    # The projection branch is used to adjust the shape of the input tensor to match the output tensor shape
    # with 1x1 convolutions
    projection = (stride != 1) or (c_in != c_out)    
    if projection:
        return nn.Sequential(
            nn.BatchNorm2d(c_in, **kw),
            nn.ReLU(),
            Add(branch, 
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False, **kw)),
        )
    else:
        return nn.Sequential(
            nn.BatchNorm2d(c_in, **kw),
            nn.ReLU(),
            Residual(branch),
        )

class ResNet9(nn.Module):
    def __init__(self, num_classes, c=64, **kw):
        super(ResNet9, self).__init__()
        if isinstance(c, int):
            c = [c, 2*c, 4*c, 4*c]

        self.prep = prep_block(3, c[0], **kw)
        self.layer1 = nn.Sequential(
            res_block(c[0], c[0], stride=1, **kw),
            res_block(c[0], c[0], stride=1, **kw),
        )
        # H,W: 32 -> 16, as strides
        self.layer2 = nn.Sequential(
            res_block(c[0], c[1], stride=2, **kw),
            res_block(c[1], c[1], stride=1, **kw),
        )
        # H,W: 16 -> 8
        self.layer3 = nn.Sequential(
            res_block(c[1], c[2], stride=2, **kw),
            res_block(c[2], c[2], stride=1, **kw),
        )
        # H,W: 8 -> 4
        self.layer4 = nn.Sequential(
            res_block(c[2], c[3], stride=2, **kw),
            res_block(c[3], c[3], stride=1, **kw),
        )
        # H,W: 4 -> 1
        self.pool = nn.MaxPool2d(kernel_size=4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=c[3], out_features=num_classes, **kw),
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

#%%
# from torchviz import make_dot
# model = ResNet9(10) # x = torch.randn(1, 3, 32, 32)
# y = model(x)
# make_dot(y, params=dict(model.named_parameters()))

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def train_cifar10(
    model,
    train_dataloader,
    test_dataloader,
    device=None,
    n_epochs=None,
    optimizer=None,
    scheduler=None,
    callback=None,
):
    # Define the criterion and optimizer
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
        test_loss, test_acc = eval_epoch(model, test_dataloader, criterion, device)
        if callback is not None:
            callback(epoch, train_acc, train_loss, test_acc, test_loss)
        if scheduler:
            scheduler.step()

def generate_options(grid):
    keys, values = zip(*grid.items())
    options = [dict(zip(keys, v)) for v in product(*values)]
    return options

def log_progress(i, freq, train_acc, train_loss, test_acc, test_loss):
    if i % freq == 0:
        logger.info(f"Epoch {i}: train acc: {train_acc}, train loss: {train_loss}")
        logger.info(f"Epoch {i}: test acc: {test_acc}, test loss: {test_loss}")
        wandb.log({
            'epoch': i,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc': test_acc,
            'test_loss': test_loss
        })

def run_experiment_with(exp_name, model, train_dataloader, test_dataloader, cfg):
    def callback(epoch, train_acc, train_loss, test_acc, test_loss):
        log_progress(epoch, cfg["log_freq"], train_acc, train_loss, test_acc, test_loss)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run = wandb.init(project=exp_name, config=cfg)
    optimizer_cls = torch.optim.__dict__[cfg["optimizer"]]
    opt = optimizer_cls(model.parameters(), 
                              lr=cfg["lr"], 
                              weight_decay=cfg["weight_decay"],
                              momentum=cfg["momentum"])

    # Baseline 1 with a manual learning rate schedule
    scheduler1 = LinearLR(opt, start_factor=0.01, end_factor=0.1, total_iters=15)
    scheduler2 = LinearLR(opt, start_factor=0.1, end_factor=0.01, total_iters=15)
    scheduler3 = LinearLR(opt, start_factor=0.01, end_factor=0.001, total_iters=5)
    scheduler = SequentialLR(opt, schedulers=[
        scheduler1, scheduler2, scheduler3], milestones=[15, 30])

    train_cifar10(
        model,
        train_dataloader,
        test_dataloader,
        device=device,
        n_epochs=cfg["n_epochs"],
        optimizer=opt,
        scheduler=scheduler,
        callback=callback,
    )
    wandb.finish()

# %%
if __name__ == "__main__":
    # %%
    # Convenient transform to normalize the image data
    base_cfg = {
        "batch_size": 128,
        "n_epochs": 35,
        "log_freq": 1,
        "optimizer": "SGD",
        "weight_decay": 0.001,
        "lr": 0.01,
        "momentum": 0.9,
    }

    train_transform = transforms.Compose(
        [
            transforms.ColorJitter(contrast=0.5),
            # transforms.RandomResize(min_size=24, max_size=40),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=base_cfg["batch_size"],
                                              shuffle=True, num_workers=2)

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=base_cfg["batch_size"],
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # %%
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet9(10, device=device)
    cfg = base_cfg.copy()
    run_experiment_with("resnet9", model, trainloader, testloader, cfg)