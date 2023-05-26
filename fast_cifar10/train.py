#%%
import torch
import torch.nn as nn
import numpy as np
import time
from collections import OrderedDict
import itertools

import torchvision
from torchvision import transforms
import torch.optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.profiler import profile, record_function, ProfilerActivity

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

def prep_block(c_in, c_out, prep_bn_relu=True, **kw):
    if prep_bn_relu:
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw)),
            ('bn1', nn.BatchNorm2d(c_out, **kw)),
            ('relu1', nn.ReLU()),
        ]))
    else:
        return nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw)

def res_block(c_in, c_out, stride, **kw):
    branch = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False, **kw)),
        ('bn1', nn.BatchNorm2d(c_out, **kw)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw)),
    ]))
    # The projection branch is used to adjust the shape of the input tensor to match the output tensor shape
    # with 1x1 convolutions
    projection = (stride != 1) or (c_in != c_out)    
    if projection:
        return nn.Sequential(OrderedDict([
            ('bn2', nn.BatchNorm2d(c_in, **kw)),
            ('relu2', nn.ReLU()),
            ('residual_conv3', Add(branch, 
                        nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False, **kw))),
        ]))
    else:
        return nn.Sequential(OrderedDict([
            ('bn2', nn.BatchNorm2d(c_in, **kw)),
            ('relu2', nn.ReLU()),
            ('residual', Residual(branch)),
        ]))

def shortcut_block(c_in, c_out, stride, **kw):
    blocks = [
        ('bn1', nn.BatchNorm2d(c_in, **kw)),
        ('relu1', nn.ReLU()),
    ]
    projection = (stride != 1) or (c_in != c_out)    
    if projection:
        blocks.append(
            ('conv3', nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False, **kw)))
    return nn.Sequential(OrderedDict(blocks))

def maxpool_block(c_in, c_out, extra_conv=False, extra_res=False, **kw):
    blocks = [
        ('conv1', nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw)),
        ('bn1', nn.BatchNorm2d(c_out, **kw)),
        ('relu1', nn.ReLU()),
        ('maxpool1', nn.MaxPool2d(2))
    ]
    return nn.Sequential(OrderedDict(blocks))

def flexible_block(c_in, c_out, extra_conv=False, extra_res=False, **kw):
    logger.info(f"flexible_block: c_in={c_in}, c_out={c_out}, extra_conv={extra_conv}, extra_res={extra_res}")
    blocks = [
        ('conv1', nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw)),
        ('bn1', nn.BatchNorm2d(c_out, **kw)),
        ('relu1', nn.ReLU()),
        ('maxpool1', nn.MaxPool2d(2))
    ]
    if extra_conv:
        blocks += [
            ('conv2', nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw)),
            ('bn2', nn.BatchNorm2d(c_out, **kw)),
            ('relu2', nn.ReLU()),
        ]

    branch = nn.Sequential(OrderedDict(blocks))
    if extra_res:
        extra_blocks = [
            ('branch', branch),
            ('conv3', nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw)),
            ('bn3', nn.BatchNorm2d(c_out, **kw)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False, **kw)),
            ('bn4', nn.BatchNorm2d(c_out, **kw)),
            ('relu4', nn.ReLU()),
        ]
        return Add(branch, nn.Sequential(OrderedDict(extra_blocks)))
    else:
        return branch

class ResNet9(nn.Module):
    def __init__(self, num_classes, block, c=64, arch_extra_convs=(), arch_extra_residuals=(), **kw):
        super(ResNet9, self).__init__()
        if isinstance(c, int):
            c = [c, 2*c, 4*c, 4*c]

        self.prep = prep_block(3, c[0], prep_bn_relu=True, **kw)

        # H,W: 32 -> 16, as strides
        self.layer1 = nn.Sequential(
            block(c[0], c[1], 
                  extra_conv = (1 in arch_extra_convs), 
                  extra_res = (1 in arch_extra_residuals),
                  **kw),
        )
        # H,W: 16 -> 8
        self.layer2 = nn.Sequential(
            block(c[1], c[2], 
                  extra_conv = (2 in arch_extra_convs), 
                  extra_res = (2 in arch_extra_residuals),
                  **kw),
        )
        # H,W: 8 -> 4
        self.layer3 = nn.Sequential(
            block(c[2], c[3], 
                  extra_conv = (3 in arch_extra_convs), 
                  extra_res = (3 in arch_extra_residuals),
                  **kw),
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
        loss = criterion(outputs.float(), labels.long())  # cross entropy loss only works with 4 bytes

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
            loss = criterion(outputs.float(), labels.long()) 

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

def chained_lr_schedule(opt, schedules):
    """
    schedules: [](start_factor, end_factor, total_iters)
    """
    schedulers = [LinearLR(opt, start_factor=s[0], end_factor=s[1], total_iters=s[2]) for s in schedules]
    milestones = [s[2] for s in schedules]
    milestones = [sum(milestones[:i+1]) for i in range(len(milestones))]
    scheduler = SequentialLR(opt, schedulers=schedulers, milestones=milestones[:-1])
    return scheduler, milestones[-1]

def run_baseline_experiment(exp_name, trainset, testset, cfg):
    def callback(epoch, *args):
        log_progress(epoch, cfg["log_freq"], *args)

    tags = cfg["tags"]
    if cfg['float16']:
        tags.append('float16')
        
    run = wandb.init(project=exp_name, config=cfg, group='SGD', tags=tags)

    if cfg['sweep']:
        cfg.update({
            "batch_size": wandb.config.batch_size,
            "lr": wandb.config.lr,
            "weight_decay": wandb.config.weight_decay,
            "lr_schedules": wandb.config.lr_schedules,
            "arch_extra_convs": wandb.config.arch_extra_convs,
            "arch_extra_residuals": wandb.config.arch_extra_residuals,
        })

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if cfg["float16"]:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = ResNet9(10, 
                    c=cfg["arch_c"],
                    block=flexible_block,
                    arch_extra_convs=cfg["arch_extra_convs"],
                    arch_extra_residuals=cfg["arch_extra_residuals"],
                    device=device, 
                    dtype=dtype)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg["batch_size"],
                                              shuffle=True, 
                                              num_workers=4)

    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg["batch_size"],
                                            shuffle=False, 
                                            num_workers=4)

    opt = torch.optim.SGD(model.parameters(),
                          lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"],
                          momentum=cfg["momentum"])

    scheduler, n_epochs = chained_lr_schedule(opt, cfg['lr_schedules'])
    print(f"Using epochs: {n_epochs} with lr_schedules: {cfg['lr_schedules']}")

    train_cifar10(
        model,
        trainloader,
        testloader,
        device=device,
        n_epochs=n_epochs,
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
        "batch_size": 256,
        "lr_schedules": [
            (1e-3, 0.075, 8),
            (0.075, 0.001, 12),
        ],
        "lr": 4,
        "log_freq": 1,
        "optimizer": "SGD",
        "weight_decay": 0.001, # 5e-4
        "momentum": 0.9,
        "float16": True,
        "sweep": True,
        "tags": ["backbone"],
        "arch_c": [64, 128, 256, 512],
        "arch_extra_convs": (),
        "arch_extra_residuals": (),
    }

    train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.Cutout(n_holes=1, length=8),
        ]
    all_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
    ]
    if base_cfg['float16']:
        all_transforms += [transforms.Lambda(lambda x: x.half())]

    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True,
                                            download=True, 
                                            transform=transforms.Compose(train_transforms + all_transforms))
    testset = torchvision.datasets.CIFAR10(root='./data', 
                                           train=False,
                                           download=True, 
                                           transform=transforms.Compose(all_transforms))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # %%
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Single run
    cfg = base_cfg.copy()
    try:
        run_baseline_experiment("resnet9", trainset, testset, cfg)
    except:
        import pdb; pdb.post_mortem()
    exit(0)

    # # Sweep run
    def generate_subsets(sequence):
        subsets = []
        for length in range(len(sequence) + 1):
            subsets.extend(itertools.combinations(sequence, length))
        return subsets

    sweep_configuration = {
        'method': 'random',
        'name': 'arch_search',
        'metric': {'goal': 'maximize', 'name': 'test_acc'},
        'parameters': 
        {
            "arch_extra_convs": {'values': generate_subsets([1,2,3])},
            "arch_extra_residuals": {'values': generate_subsets([1,2,3])},
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project='resnet9'
    )

    cfg = base_cfg.copy()
    def launch_sweep():
        if cfg['float16']:
            model = ResNet9(10, device=device, dtype=torch.float16)
        else:
            model = ResNet9(10, device=device)
        run_baseline_experiment("resnet9", model, trainset, testset, cfg)

    wandb.agent(sweep_id, launch_sweep, count=4)

    # %% Profile run
    # cfg = base_cfg.copy()
    # cfg.update({
    #     'batch_size': 128,
    #     'n_epochs': 1,
    # })
    # input = torch.randn(cfg['batch_size'], 3, 32, 32).to(device)

    # with profile(activities=[
    #         ProfilerActivity.CUDA], 
    #              record_shapes=True, 
    #              with_modules=True,
    #              with_flops=True,
    #              ) as prof:
    #     with record_function("model_inference"):
    #         for i in range(cfg['n_epochs']):
    #             run_baseline_experiment("resnet9", model, trainset, testset, cfg)

#%%
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=5))
# %%
