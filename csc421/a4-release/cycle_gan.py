#!/usr/bin/env python
# coding: utf-8

# # Colab FAQ
# 
# For some basic overview and features offered in Colab notebooks, check out: [Overview of Colaboratory Features](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
# 
# You need to use the colab GPU for this assignmentby selecting:
# 
# > **Runtime**   →   **Change runtime type**   →   **Hardware Accelerator: GPU**

# ## Setup PyTorch
# All files are stored at /content/csc421/a4/ folder
# 

# In[ ]:


######################################################################
# Setup python environment and change the current working directory
######################################################################
# !pip install torch torchvision
# !pip install Pillow==4.0.0


# # Helper code

# ## Utility functions

# In[ ]:


import os
import pdb
import argparse
import pickle as pkl

from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from six.moves.urllib.request import urlretrieve
import tarfile
import pickle
import sys
import scipy
import imageio

def get_file(fname,
             origin,
             untar=False,
             extract=False,
             archive_format='auto',
             cache_dir='data'):
    datadir = os.path.join(cache_dir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)
    
    print(fpath)
    if not os.path.exists(fpath):
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if untar:
        if not os.path.exists(untar_fpath):
            print('Extracting file.')
            with tarfile.open(fpath) as archive:
                archive.extractall(datadir)
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def to_var(tensor):
    """Wraps a Tensor in a Variable, optionally placing it on the GPU.

        Arguments:
            tensor: A Tensor object.
            cuda: A boolean flag indicating whether to use the GPU.

        Returns:
            A Variable object, on the GPU if cuda==True.
    """
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def create_dir(directory):
    """Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def gan_checkpoint(iteration, G, D, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G.pkl')
    D_path = os.path.join(opts.checkpoint_dir, 'D.pkl')
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)

def cyclegan_checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    G_XtoY_path = os.path.join(opts.checkpoint_dir, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.checkpoint_dir, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.checkpoint_dir, 'D_X.pkl')
    D_Y_path = os.path.join(opts.checkpoint_dir, 'D_Y.pkl')
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints.
    """
    G_XtoY_path = os.path.join(opts.load, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.load, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.load, 'D_X.pkl')
    D_Y_path = os.path.join(opts.load, 'D_Y.pkl')

    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    G_XtoY.load_state_dict(torch.load(G_XtoY_path, map_location=lambda storage, loc: storage))
    G_YtoX.load_state_dict(torch.load(G_YtoX_path, map_location=lambda storage, loc: storage))
    D_X.load_state_dict(torch.load(D_X_path, map_location=lambda storage, loc: storage))
    D_Y.load_state_dict(torch.load(D_Y_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y


def merge_images(sources, targets, opts):
    """Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    _, _, h, w = sources.shape
    row = int(np.sqrt(opts.batch_size))
    merged = np.zeros([3, row*h, row*w*2])
    for (idx, s, t) in (zip(range(row**2), sources, targets,)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    return merged.transpose(1, 2, 0)

def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape
    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h*nrows, cell_w*ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = array[i*ncols+j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result

def save_real_samples(images, batch_index, opts):
    images = denormalize(opts.X, images)
    images = to_data(images)
    grid = create_image_grid(images)
    path = os.path.join(opts.sample_dir, 'real-image-{:06d}.png'.format(batch_index))
    imageio.imsave(path, (grid * 255).astype(np.uint8))


def gan_save_samples(G, fixed_noise, iteration, opts):
    generated_images = denormalize(opts.X, G(fixed_noise).detach())
    generated_images = to_data(generated_images)

    grid = create_image_grid(generated_images)

    path = os.path.join(opts.sample_dir, 'sample-{:06d}.jpg'.format(iteration))
    imageio.imsave(path, (grid * 255).astype(np.uint8))

def cyclegan_save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    fake_X = G_YtoX(fixed_Y)
    fake_Y = G_XtoY(fixed_X)
    fake_X = denormalize(opts.X, fake_X)
    fake_Y = denormalize(opts.Y, fake_Y)

    fixed_X = denormalize(opts.X, fixed_X)
    fixed_Y = denormalize(opts.Y, fixed_Y)

    X, fake_X = to_data(fixed_X), to_data(fake_X)
    Y, fake_Y = to_data(fixed_Y), to_data(fake_Y)

    merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-X-Y.jpg'.format(iteration))
    imageio.imsave(path, (merged * 255).astype(np.uint8))
    print('Saved {}'.format(path))

    merged = merge_images(Y, fake_X, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-Y-X.jpg'.format(iteration))
    imageio.imsave(path, (merged * 255).astype(np.uint8))
    print('Saved {}'.format(path))


# ## Data loader

# In[ ]:
def compute_emoji_mean_std(emoji_type, opts):
    """Compute the mean, std of the training dataset.
    The value would later be used to normalize the training batch.
    """
    transform = transforms.Compose([
                    transforms.Resize(opts.image_size),
                    transforms.ToTensor(),
                ])

    train_path = os.path.join('data/emojis', emoji_type)
    train_dataset = datasets.ImageFolder(train_path, transform)
    loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers)

    mean = 0.
    std = 0.

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return (mean, std)


def denormalize(emoji_type, tensor):
    """Denormalize a mini batch of image tensors according to training mean and std.

     It works with 3-or-4 dimension tensors.
    """
    if emoji_type == 'Windows':
        mean = (0.3495, 0.4029, 0.3075)
        std = (0.2539, 0.2157, 0.2143)
    else:
        mean = (0.4643, 0.4766, 0.3764)
        std = (0.2168, 0.1608, 0.1608)

    tensor = tensor.clone().detach()

    if tensor.ndim == 4: # Assume NCHW dimensions
        tensor = tensor.permute(1, 2, 3, 0)

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # inplace multiplication and addition

    if tensor.ndim == 4:
        tensor = tensor.permute(3, 0, 1, 2)

    return tensor


# In[ ]:
def get_emoji_loader(emoji_type, opts, mean=(0.3495, 0.4029, 0.3075), std=(0.2539, 0.2157, 0.2143)):
    """Creates training and test data loaders. The mean/std values are computed over training dataset."""

    if emoji_type == 'Windows':
        mean = (0.3495, 0.4029, 0.3075)
        std = (0.2539, 0.2157, 0.2143)
    else:
        mean = (0.4643, 0.4766, 0.3764)
        std = (0.2168, 0.1608, 0.1608)

    transform = transforms.Compose([
                    transforms.Resize(opts.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

    train_path = os.path.join('data/emojis', emoji_type)
    test_path = os.path.join('data/emojis', 'Test_{}'.format(emoji_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    return train_dloader, test_dloader


# ## Training and evaluation code

# In[ ]:


def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    if G_YtoX:
      print("                 G_XtoY                ")
      print("---------------------------------------")
      print(G_XtoY)
      print("---------------------------------------")

      print("                 G_YtoX                ")
      print("---------------------------------------")
      print(G_YtoX)
      print("---------------------------------------")

      print("                  D_X                  ")
      print("---------------------------------------")
      print(D_X)
      print("---------------------------------------")

      print("                  D_Y                  ")
      print("---------------------------------------")
      print(D_Y)
      print("---------------------------------------")
    else:
      print("                 G                     ")
      print("---------------------------------------")
      print(G_XtoY)
      print("---------------------------------------")

      print("                  D                    ")
      print("---------------------------------------")
      print(D_X)
      print("---------------------------------------")

      
def create_model(opts):
    """Builds the generators and discriminators.
    """
    if opts.Y is None:
      ### GAN
      G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.g_conv_dim)
      D = DCDiscriminator(conv_dim=opts.d_conv_dim)

      print_models(G, None, D, None)

      if torch.cuda.is_available():
          G.cuda()
          D.cuda()
          print('Models moved to GPU.')
      return G, D
          
    else:
      ### CycleGAN
      G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
      G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
      D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
      D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

      print_models(G_XtoY, G_YtoX, D_X, D_Y)

      if torch.cuda.is_available():
          G_XtoY.cuda()
          G_YtoX.cuda()
          D_X.cuda()
          D_Y.cuda()
          print('Models moved to GPU.')
      return G_XtoY, G_YtoX, D_X, D_Y


def train(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_emoji_loader(emoji_type=opts.X, opts=opts)
    if opts.Y:
      dataloader_Y, test_dataloader_Y = get_emoji_loader(emoji_type=opts.Y, opts=opts)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)

    # Start training
    if opts.Y is None:
      G, D = gan_training_loop(dataloader_X, test_dataloader_X, opts)
      return G, D
    else:
      G_XtoY, G_YtoX, D_X, D_Y = cyclegan_training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts)
      return G_XtoY, G_YtoX, D_X, D_Y


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)




# # Your code for generators and discriminators

# ## Helper modules

# In[ ]:


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)
  

def upconv(in_channels, out_channels, kernel_size, stride=2, padding=2, batch_norm=True):
    """Creates a upsample-and-convolution layer, with optional batch normalization.
    """
    layers = []
    if stride>1:
      layers.append(nn.Upsample(scale_factor=stride))
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=2, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
  
  
class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


# ##DCGAN

# ### GAN generator

# In[ ]:


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(DCGenerator, self).__init__()

        self.conv_dim = conv_dim
        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # NOTE: upconv, padding is padding of the transposed conv, not the original feature map. (Unlike py torch)
        # Here, we manually calculat the padding needed.
        # i*S - k + 2p + 1; p = [(o - i*S) + (k - 1)]/2
        self.linear_bn = nn.Sequential(
            nn.Linear(100, 128 * 4 * 4),
            nn.BatchNorm1d(128*4*4)
        )
        self.upconv1 = upconv(128, 64, 5, 2, 2)
        self.upconv2 = upconv(64, 32, 5, 2, 2)
        self.upconv3 = upconv(32, 3, 5, 2, 2, batch_norm=False)

    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  BSx100x1x1 (during training)

            Output
            ------
                out: BS x channels x image_width x image_height  -->  BSx3x32x32 (during training)
        """
        batch_size = z.size(0)
        
        out = F.relu(self.linear_bn(z.squeeze())).view(-1, self.conv_dim*4, 4, 4)    # BS x 128 x 4 x 4
        out = F.relu(self.upconv1(out))  # BS x 64 x 8 x 8
        out = F.relu(self.upconv2(out))  # BS x 32 x 16 x 16
        out = torch.tanh(self.upconv3(out))  # BS x 3 x 32 x 32
        
        out_size = out.size()
        if out_size != torch.Size([batch_size, 3, 32, 32]):
          raise ValueError("expect {} x 3 x 32 x 32, but get {}".format(batch_size, out_size))
        return out


# ### GAN discriminator

# In[ ]:


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64):
        super(DCDiscriminator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # (i - k + 2p)/s + 1 = i/2
        self.conv1 = conv(3, 32, 5, 2, 2)
        self.conv2 = conv(32, 64, 5, 2, 2)
        self.conv3 = conv(64, 128, 5, 2, 2)
        self.conv4 = conv(128, 1, 4, 1, 0, batch_norm=False)  # As a linear layer

    def forward(self, x):
        batch_size = x.size(0)

                                       # BS x 3 x 32 x 32
        out = F.relu(self.conv1(x))    # BS x 32 x 16 x 16
        out = F.relu(self.conv2(out))    # BS x 64 x 8 x 8
        out = F.relu(self.conv3(out))    # BS x 128 x 4 x 4

        out = self.conv4(out).squeeze()
        out_size = out.size()
        if out_size != torch.Size([batch_size,]):
          raise ValueError("expect {} x 1, but get {}".format(batch_size, out_size))
        return out


# ###GAN training loop

# In[ ]:


def gan_training_loop(dataloader, test_dataloader, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G, D = create_model(opts)

    g_params = G.parameters()  # Get generator parameters
    d_params = D.parameters()  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr*2., [opts.beta1, opts.beta2])

    train_iter = iter(dataloader)

    test_iter = iter(test_dataloader)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_noise = sample_noise(100, opts.noise_size)  # # 100 x noise_size x 1 x 1

    iter_per_epoch = len(train_iter)
    total_train_iters = opts.train_iters

    try:
      for iteration in range(1, opts.train_iters+1):

          # Reset data_iter for each epoch
          if iteration % iter_per_epoch == 0:
              train_iter = iter(dataloader)

          # NOTE: real_labels are not used, manually assign 1 or 0
          real_images, real_labels = next(train_iter)
          real_images, real_labels = to_var(real_images), to_var(real_labels).long().squeeze()

          if iteration // iter_per_epoch == 0:
            save_real_samples(real_images, iteration, opts)

          d_optimizer.zero_grad()

          m = real_images.shape[0]
          # FILL THIS IN
          # 1. Compute the discriminator loss on real images
          D_real_loss = torch.mean((D(real_images) - 1)**2)

          # 2. Sample noise
          noise_entries = torch.randint(100, (m,))  # Select 10 random entries
          noise = fixed_noise[noise_entries]

          # 3. Generate fake images from the noise
          fake_images = G(noise)

          # 4. Compute the discriminator loss on the fake images
          D_fake_loss = torch.mean((D(fake_images) ** 2))          

          # 5. Compute the total discriminator loss
          D_total_loss = (D_real_loss + D_fake_loss)/2.0

          D_total_loss.backward()
          d_optimizer.step()

          ###########################################
          ###          TRAIN THE GENERATOR        ###
          ###########################################

          g_optimizer.zero_grad()

          # FILL THIS IN
          # 1. Sample noise
          noise_entries = torch.randint(100, (m,))  # Select 10 random entries
          noise = fixed_noise[noise_entries]

          # 2. Generate fake images from the noise
          fake_images = G(noise)

          # 3. Compute the generator loss
          G_loss = torch.mean((D(fake_images) - 1)**2)  # TODO: Do I need to manually turn 1 into a tensor?
          
          G_loss.backward()
          g_optimizer.step()


          # Print the log info
          if iteration % opts.log_step == 0:
              print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                     iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))

          # Save the generated samples
          if iteration % opts.sample_every == 0:
              gan_save_samples(G, fixed_noise, iteration, opts)

          # Save the model parameters
          if iteration % opts.checkpoint_every == 0:
              gan_checkpoint(iteration, G, D, opts)
              
    except KeyboardInterrupt:
        print('Exiting early from training.')
        return G, D
      
    return G, D




# ##CycleGAN

# ###CycleGAN generator

# In[ ]:


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(3, 32, 5, 2, 2)
        self.conv2 = conv(32, 64, 5, 2, 2)

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(64)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.upconv1 = upconv(64, 32, 5, 2, 2)
        self.upconv2 = upconv(32, 3, 5, 2, 2)


    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """
        batch_size = x.size(0)
        
        out = F.relu(self.conv1(x))            # BS x 32 x 16 x 16
        out = F.relu(self.conv2(out))          # BS x 64 x 8 x 8
        
        out = F.relu(self.resnet_block(out))   # BS x 64 x 8 x 8

        out = F.relu(self.upconv1(out))        # BS x 32 x 16 x 16
        out = torch.tanh(self.upconv2(out))        # BS x 3 x 32 x 32
        
        out_size = out.size()
        if out_size != torch.Size([batch_size, 3, 32, 32]):
          raise ValueError("expect {} x 3 x 32 x 32, but get {}".format(batch_size, out_size))


        return out


# ###CycleGAN training loop

# In[ ]:


def cyclegan_training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G_XtoY, G_YtoX, D_X, D_Y = create_model(opts)

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
    d_params = list(D_X.parameters()) + list(D_Y.parameters())  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = to_var(next(test_iter_X)[0])
    fixed_Y = to_var(next(test_iter_Y)[0])

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    try:
      for iteration in range(1, opts.train_iters+1):

          # Reset data_iter for each epoch
          if iteration % iter_per_epoch == 0:
              iter_X = iter(dataloader_X)
              iter_Y = iter(dataloader_Y)

          images_X, labels_X = next(iter_X)
          images_X, labels_X = to_var(images_X), to_var(labels_X).long().squeeze()

          images_Y, labels_Y = next(iter_Y)
          images_Y, labels_Y = to_var(images_Y), to_var(labels_Y).long().squeeze()


          # ============================================
          #            TRAIN THE DISCRIMINATORS
          # ============================================

          #########################################
          ##             FILL THIS IN            ##
          #########################################

          # Train with real images
          d_optimizer.zero_grad()

          # 1. Compute the discriminator losses on real images
          D_X_loss = torch.mean((D_X(images_X) - 1) ** 2)
          D_Y_loss = torch.mean((D_Y(images_Y) - 1) ** 2)
          
          d_real_loss = D_X_loss + D_Y_loss
          d_real_loss.backward()
          d_optimizer.step()

          # Train with fake images
          d_optimizer.zero_grad()

          # 2. Generate fake images that look like domain X based on real images in domain Y
          fake_X = G_YtoX(images_Y)

          # 3. Compute the loss for D_X
          D_X_loss = torch.mean(D_X(fake_X) ** 2)

          # 4. Generate fake images that look like domain Y based on real images in domain X
          fake_Y = G_XtoY(images_X)

          # 5. Compute the loss for D_Y
          D_Y_loss = torch.mean(D_Y(fake_Y) ** 2)

          d_fake_loss = D_X_loss + D_Y_loss
          d_fake_loss.backward()
          d_optimizer.step()


          # =========================================
          #            TRAIN THE GENERATORS
          # =========================================


          #########################################
          ##    FILL THIS IN: Y--X-->Y CYCLE     ##
          #########################################
          g_optimizer.zero_grad()

          # 1. Generate fake images that look like domain X based on real images in domain Y
          fake_X = G_YtoX(images_Y)

          # 2. Compute the generator loss based on domain X
          g_loss = torch.mean((D_X(fake_X) - 1) ** 2)

          reconstructed_Y = G_XtoY(fake_X)
          # 3. Compute the cycle consistency loss (the reconstruction loss)
          cycle_consistency_loss = torch.mean(reconstructed_Y - images_Y)
          
          g_loss += opts.lambda_cycle * cycle_consistency_loss
          
          g_loss.backward()
          g_optimizer.step()



          #########################################
          ##    FILL THIS IN: X--Y-->X CYCLE     ##
          #########################################

          g_optimizer.zero_grad()

          # 1. Generate fake images that look like domain Y based on real images in domain X
          fake_Y = G_XtoY(images_X)

          # 2. Compute the generator loss based on domain Y
          g_loss = torch.mean((D_Y(fake_Y) - 1) ** 2)

          reconstructed_X = G_YtoX(fake_Y)
          # 3. Compute the cycle consistency loss (the reconstruction loss)
          cycle_consistency_loss = torch.mean(reconstructed_X - images_X)
          
          g_loss += opts.lambda_cycle * cycle_consistency_loss

          g_loss.backward()
          g_optimizer.step()


          # Print the log info
          if iteration % opts.log_step == 0:
              print('Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | '
                    'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                      iteration, opts.train_iters, d_real_loss.item(), D_Y_loss.item(),
                      D_X_loss.item(), d_fake_loss.item(), g_loss.item()))


          # Save the generated samples
          if iteration % opts.sample_every == 0:
              cyclegan_save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)


          # Save the model parameters
          if iteration % opts.checkpoint_every == 0:
              cyclegan_checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)

    except KeyboardInterrupt:
        print('Exiting early from training.')
        return G_XtoY, G_YtoX, D_X, D_Y
      
    return G_XtoY, G_YtoX, D_X, D_Y


# # Training
# 

# ## Download dataset

# In[ ]:


######################################################################
# Download Translation datasets
######################################################################
data_fpath = get_file(fname='emojis', 
                         origin='http://www.cs.toronto.edu/~jba/emojis.tar.gz', 
                         untar=True)


# %% helper function to display the image
def tensor_to_display(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor.transpose(1, 2, 0)
    elif isinstance(tensor, torch.Tensor):
        return tensor.permute(1, 2, 0)
    else:
        raise TypeError("Unsupported type for tensor_to_display function")


def display_batch(batch, row=6, col=6):
    fig, axs = plt.subplots(row, col, figsize=(10, 10))
    for i in range(row):
        for j in range(col):
            if i*row+j < batch.shape[0]:
                axs[i, j].imshow(tensor_to_display(batch[i*row+j]))
                axs[i, j].axis('off')
    plt.show()

# ## DCGAN

# In[ ]:


SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


args = AttrDict()
args_dict = {
              'image_size':32, 
              'g_conv_dim':32, 
              'd_conv_dim':64,
              'noise_size':100,
              'num_workers': 0,
              'train_iters':5000,
              'X':'Windows',  # options: 'Windows' / 'Apple'
              'Y': None,
              'lr':0.0003,
              'beta1':0.5,
              'beta2':0.999,
              'batch_size':32, 
              'checkpoint_dir': 'checkpoints_gan',
              'sample_dir': 'samples_gan',
              'load': None,
              'log_step':200,
              'sample_every':200,
              'checkpoint_every':1000,
}
args.update(args_dict)
print_opts(args)

# %%
# Sample the data as 32x32 image and plot the image
opts = args
dataloader_X, test_dataloader_X = get_emoji_loader(emoji_type="Apple", opts=opts)
sample = next(iter(dataloader_X))[0]
# display_batch(denormalize(opts.X, sample))

# %% Training DCGAN
# G, D = train(args)

# ## CycleGAN

# In[ ]:


SEED = 4

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


args = AttrDict()
args_dict = {
              'image_size':32, 
              'g_conv_dim':32, 
              'd_conv_dim':32,
              'init_zero_weights': False,
              'num_workers': 0,
              'train_iters':5000,
              'X':'Apple',
              'Y':'Windows',
              'lambda_cycle': 0.015,
              'lr':0.0003,
              'beta1':0.3,
              'beta2':0.999,
              'batch_size':32, 
              'checkpoint_dir': 'checkpoints_cyclegan',
              'sample_dir': 'samples_cyclegan',
              'load': None,
              'log_step':200,
              'sample_every':200,
              'checkpoint_every':1000,
}
args.update(args_dict)


print_opts(args)
G_XtoY, G_YtoX, D_X, D_Y = train(args)


# In[ ]:
