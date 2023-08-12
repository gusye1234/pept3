import pytest

from pept3 import models


def test_deps():
    import h5py
    import numpy
    import pandas
    import torch


def test_checkpoints():
    import hashlib

    md5_book = models._Model_Weights_md5_Factories
    for name, md5_v in md5_book.items():
        weights = models.Model_Weights_Factories(name)
        weights_bin = open(weights, 'rb')
        md5_test = hashlib.md5()
        md5_test.update(weights_bin.read())
        assert md5_test.hexdigest() == md5_v
