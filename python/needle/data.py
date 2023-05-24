import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any, Tuple
from needle import backend_ndarray as nd

import gzip
import struct
from scipy.ndimage import zoom


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            img = np.flip(img, axis = 1)
        return img

class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        padding = self.padding
        h, w, c = img.shape
        shift_x, shift_y = np.random.randint(low=-padding, high=padding+1, size=2)
        img = np.pad(img, ((padding,padding),(padding,padding),(0,0)))
        # The old origin now sits at (padding,padding)
        img_padded = img[
            padding+shift_x:padding+shift_x+h,
            padding+shift_y:padding+shift_y+w,
            :]
        assert img_padded.shape == (h, w, c), "Padding should preserve the shape"
        return img_padded

class RandomScale:
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Randomly scale an image preserving the original shape.

        Args:
            img (np.ndarray): Original image in the shape (C, H, W).

        Returns:
            np.ndarray: Scaled image in the shape (C, H, W).
        """
        scale_factor = np.random.uniform(low=self.scale_range[0], high=self.scale_range[1])
        c, h, w = img.shape  # Fix this line

        # Use scipy's zoom function for resizing
        img_resized = zoom(img, (1, scale_factor, scale_factor))

        new_c, new_h, new_w = img_resized.shape  # Fix this line

        # If the scale factor is less than 1, we need to pad the image
        if scale_factor < 1:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img_padded = np.pad(img_resized, ((0, 0), (pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w)), mode='constant')
        # If the scale factor is greater than 1, we need to crop the image
        else:
            crop_h = (new_h - h) // 2
            crop_w = (new_w - w) // 2
            img_padded = img_resized[:, crop_h:crop_h+h, crop_w:crop_w+w]

        assert img_padded.shape == img.shape, "Scaling should preserve the original shape"
        return img_padded


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if hasattr(self, 'transforms') and self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device=None,
        dtype="float32"
    ) -> Tensor :

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                            range(batch_size, len(dataset), batch_size))

    def get_batch(self, batch_indices):
        return tuple(zip(*[self.dataset[i] for i in batch_indices]))

    def __iter__(self):
        if self.shuffle:
            self.batch_ordering = np.array_split(
                np.random.permutation(len(self.dataset)), len(self.dataset) // self.batch_size)
        else:
            self.batch_ordering = self.ordering
        self.batch_index = 0
        assert self.batch_ordering is not None
        return self

    def __next__(self) -> List[Tensor]:
        if self.batch_index == len(self.batch_ordering):
            raise StopIteration

        batch_indices = self.batch_ordering[self.batch_index]
        # NOTE: It expects a tensor as returned type
        batch_data = [Tensor(batch, requires_grad=False, device=self.device, dtype=self.dtype) for batch in self.get_batch(batch_indices)]
        self.batch_index += 1
        return batch_data

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        """
        Creates a MNIST dataset instance given a path to the relevant files.

        Args:
            image_filename: path to the MNIST image file
            label_filename: path to the MNIST label file
            transforms: List of transforms to be applied to the images. Refer to `Transforms` folder.

        """
        self.transforms = transforms
        images, labels = parse_mnist(image_filename, label_filename)
        assert len(images) == len(labels), "The number of images must equal the number of labels"
        # Convert the every image into (H W C) cube
        self.images = images.reshape(-1, 28, 28, 1)
        self.labels = labels

    def __getitem__(self, index) -> object:
        """
        Fetches the i-th data entry from the MNIST dataset.

        Returns:
            Tuple of two numpy arrays, where the first array corresponds to the image and the second array
            corresponds to the label.
        """
        x = self.images[index]
        y = self.labels[index]
        return self.apply_transforms(x), y

    def __len__(self) -> int:
        """
        Returns:
            The length of the MNIST dataset.
        """
        return len(self.images)


# Copied from hw1
def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.

    Notes:
        This MNIST class assumes that the image and label files are idx3 formatted, and that
        they are stored in the format of:
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  10000            number of items
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
                    ........
        xxxx     unsigned byte   ??               pixel

        The files can be downloaded from here:

        http://yann.lecun.com/exdb/mnist/

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


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_cifar_10(batch_files):
    """Read cifar10 files from base_folder, and return arrays of images and labels.
    """
    X = []
    y = []
    for filename in batch_files:
        data = unpickle(filename)
        X.append(data[b'data'])
        y += data[b'labels']
    # CIFAR is in C,H,W
    X = np.vstack(X).reshape(-1, 3, 32, 32)
    y = np.array(y)
    return X / 255.0, y


def read_cifar_10_meta(batch_meta):
    """Read cifar10 files from base_folder, and return arrays of images and labels.
    """
    data = unpickle(batch_meta)
    return data[b'label_names']


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        normalized: bool = False,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        training_files = [os.path.join(base_folder, f'data_batch_{batch}') for batch in range(1, 6)]
        test_files = [os.path.join(base_folder, 'test_batch')]
        self.X, self.Y = read_cifar_10(training_files if train else test_files)
        assert len(self.X) == len(self.Y), "Number of images and labels must match"
        assert (self.X >= 0.0).all() and (self.X <= 1.0).all(), "All entries of self.X must be between 0.0 and 1.0"

        self.normalized = normalized
        self.X_mean = np.mean(self.X, axis=(0, 2, 3)).reshape(-1, 3, 1, 1)
        self.X_std = np.std(self.X, axis=(0, 2, 3)).reshape(-1, 3, 1, 1)

    def normalize(self, x):
        return (x - self.X_mean) / self.X_std

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        x = self.X[index]
        if self.normalized: 
            x = self.normalize(x)
        y = self.Y[index]
        return self.apply_transforms(x), y

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return len(self.X)


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION