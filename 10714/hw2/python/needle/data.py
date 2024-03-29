import struct
import gzip
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


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
    ) -> Tensor :

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
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
        batch_data = [Tensor(batch, requires_grad=False) for batch in self.get_batch(batch_indices)]
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

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])

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
