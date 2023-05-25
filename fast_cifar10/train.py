#%%
import torch
import torch.nn as nn
import numpy as np
import time

import torchvision
from torchvision import transforms
import torch.optim
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
    return nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw)

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
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
        train_finished = time.time()
        test_loss, test_acc = eval_epoch(model, test_dataloader, criterion, device)
        test_finished  = time.time()

        train_elapsed_time = train_finished - epoch_start_time  # train time
        test_elapsed_time = test_finished - train_finished

        lr = optimizer.param_groups[0]['lr']

        if scheduler:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        if callback is not None:
            callback(epoch, lr, train_acc, train_loss, test_acc, test_loss, train_elapsed_time, test_elapsed_time)


def generate_options(grid):
    keys, values = zip(*grid.items())
    options = [dict(zip(keys, v)) for v in product(*values)]
    return options

def log_progress(i, freq, lr, train_acc, train_loss, test_acc, test_loss, train_elapsed_time, test_elapsed_time):
    if i % freq == 0:
        logger.info(f"Epoch {i}: train acc: {train_acc}, train loss: {train_loss}")
        logger.info(f"Epoch {i}: test acc: {test_acc}, test loss: {test_loss}")
        logger.info(f"Epoch {i}: train time: {train_elapsed_time:.2f}, test time: {test_elapsed_time:.2f}")
        logger.info(f"Epoch {i}: lr: {lr:.6f}")
        wandb.log({
            'epoch': i,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'train_elapsed_time': train_elapsed_time,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'test_elapsed_time': test_elapsed_time,
        })

def run_baseline_experiment(exp_name, model, train_dataloader, test_dataloader):
    def callback(epoch, *args):
        log_progress(epoch, cfg["log_freq"], *args)

    cfg = {
        "batch_size": 128,
        "n_epochs": 35,
        "log_freq": 1,
        "optimizer": "SGD",
        # "weight_decay": 5e-4*128,  # 5e-4 * batch_size 
        "weight_decay": 0.001,
        "lr": 0.001,
        "momentum": 0.9,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run = wandb.init(project=exp_name, config=cfg)
    opt = torch.optim.SGD(model.parameters(),
                          lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"],
                          momentum=cfg["momentum"])

    # Baseline 1 with a manual learning rate schedule
    scheduler1 = LinearLR(opt, start_factor=1, end_factor=7.5, total_iters=15)
    scheduler2 = LinearLR(opt, start_factor=7.5, end_factor=5, total_iters=15)
    scheduler3 = LinearLR(opt, start_factor=5.0, end_factor=1, total_iters=5)
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

def run_adam_experiment(exp_name, model, train_dataloader, test_dataloader):
    def callback(epoch, *args):
        log_progress(epoch, cfg["log_freq"], *args)

    cfg = {
        "batch_size": 128,
        "n_epochs": 35,
        "log_freq": 1,
        "optimizer": "Adam",
        # "weight_decay": 5e-4*128,  # 5e-4 * batch_size 
        "weight_decay": 0.001,
        "lr": 0.001,
        "momentum": 0.9,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run = wandb.init(project=exp_name, config=cfg, group='Adam', tags=[])
    opt = torch.optim.Adam(model.parameters(),
                          lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"])

    # Baseline 1 with a manual learning rate schedule
    scheduler1 = LinearLR(opt, start_factor=1, end_factor=7.5, total_iters=15)
    scheduler2 = LinearLR(opt, start_factor=7.5, end_factor=5, total_iters=15)
    scheduler3 = LinearLR(opt, start_factor=5.0, end_factor=1, total_iters=5)
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
        "lr": 1,
        "momentum": 0.9,
    }

    train_transform = transforms.Compose(
        [
            # transforms.ColorJitter(contrast=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(30),
            transforms.ToTensor(),
            # values specific for CIFAR10
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=base_cfg["batch_size"],
                                              shuffle=True, num_workers=torch.get_num_threads())
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         # values specific for CIFAR10
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
         ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=base_cfg["batch_size"],
                                            shuffle=False, num_workers=torch.get_num_threads())
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # %%
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet9(10, device=device)

    run_adam_experiment("resnet9", model, trainloader, testloader)