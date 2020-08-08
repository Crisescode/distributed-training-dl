import numpy
import os
import chainer
from chainer.dataset import download
from mnist_helper import make_npz
from mnist_helper import preprocess_mnist

def get_mnist(withlabel=True, ndim=1, scale=1, dtype=None,
              label_dtype=numpy.int32, rgb_format=False):

    dtype = chainer.get_dtype(dtype)
    train_raw = _retrieve_mnist_training()
    train = preprocess_mnist(train_raw, withlabel, ndim, scale, dtype,
                             label_dtype, rgb_format)

    test_raw = _retrieve_mnist_test()
    test = preprocess_mnist(test_raw, withlabel, ndim, scale, dtype,
                             label_dtype, rgb_format)
    return train, test

def _retrieve_mnist_training():
    train_path1 = os.path.dirname(os.path.realpath(__file__)) + '/../../datasets/mnist/train-images-idx3-ubyte.gz'
    train_path2 = os.path.dirname(os.path.realpath(__file__)) + '/../../datasets/mnist/train-labels-idx1-ubyte.gz'
    train_path = [train_path1, train_path2]
    return _retrieve_mnist('train.npz', train_path)

def _retrieve_mnist_test():
    test_path1 = os.path.dirname(os.path.realpath(__file__)) + '/../../datasets/mnist/t10k-images-idx3-ubyte.gz'
    test_path2 = os.path.dirname(os.path.realpath(__file__)) + '/../../datasets/mnist/t10k-labels-idx1-ubyte.gz'
    test_path = [test_path1, test_path2]
    return _retrieve_mnist('test.npz', test_path)

def _retrieve_mnist(name, data_paths):

    root = download.get_dataset_directory('./temp_dir')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: make_npz(path, data_paths), numpy.load)
