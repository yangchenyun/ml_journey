import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl



def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    batch_size = Z.shape[0]
    Z_y = (Z * y_one_hot).sum(axes=1)
    delta = Z.exp().sum(axes=1).log() - Z_y
    # print(f"delta.shape: {delta.shape}, delta.sum: {type(delta.sum()/batch_size)}")
    return delta.sum() / batch_size


def one_hot(input, num_classes):
    return [[1 if j == value else 0
             for j in range(num_classes)]
            for value in input]


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    m = X.shape[0]
    num_classes = W2.shape[1]
    for i in range(0, m, batch):
        if i == 0: print()
        print("\r\x1b[A\x1b[K" + f"processing batch: {100*i/m:.2f}%")

        B = np.arange(i, min(i + batch, m))
        X_b = X[B, :] # (b, n)
        y_b = y[B]    # (b, )

        # Enter the Tensor world
        X_b = ndl.Tensor(X_b, dtype="float64")
        y_h = ndl.Tensor(one_hot(y_b, num_classes), dtype="int8")

        Z_1 = (X_b @ W1).relu() # (b, d)
        Z = Z_1 @ W2
        assert Z.shape[0] == y_h.shape[0], f"Z:{Z.shape}, y_h: {y_h.shape}"
        loss = softmax_loss(Z, y_h)
        loss.backward()

        # NOTE: convert back to numpy to update, otherwise, 
        # the computational graph would blow up the memory
        W1 = ndl.Tensor(W1.numpy() - lr*(W1.grad.numpy()), dtype="float64")
        W2 = ndl.Tensor(W2.numpy() - lr*(W2.grad.numpy()), dtype="float64")

    return W1, W2

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # Read from gzipped image and label file in MNIST format
    gzip_images = gzip.open(image_filename, 'rb')
    gzip_labels = gzip.open(label_filename, 'rb')

    # File is in MSB (high endian) format, read in 4-byte integers
    # The > character indicates that the byte order should be "big-endian", 
    # which means that the most significant byte comes first. 
    # The I character specifies that the value should be interpreted as an 
    # unsigned integer.

    read_32bit_int = lambda f: struct.unpack('>I', f.read(4))[0]
    read_8bit_int = lambda f: struct.unpack('>B', f.read(1))[0]

    magic_number = read_32bit_int(gzip_images)
    if magic_number != 2051:
        raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                         (magic_number, image_filename))
    magic_number = read_32bit_int(gzip_labels)
    if magic_number != 2049:
        raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                         (magic_number, label_filename))
    # Read number of images
    num_images = read_32bit_int(gzip_images)
    num_labels = read_32bit_int(gzip_labels)
    if num_images != num_labels:
        raise ValueError('Image file and label file contain different number of images')
    # Read number of rows and columns
    num_rows = read_32bit_int(gzip_images)
    num_cols = read_32bit_int(gzip_images)
    # Read images and labels
    images = np.zeros((num_images, num_rows * num_cols), dtype=np.float32)
    labels = np.zeros(num_images, dtype=np.uint8)
    for i in range(num_images):
        for j in range(num_rows * num_cols):
            images[i, j] = read_8bit_int(gzip_images)
        labels[i] = read_8bit_int(gzip_labels)
    # Normalize images
    images = images / 255.0
    return images, labels


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
