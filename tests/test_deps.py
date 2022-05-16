import pytest

from stums import models


def test_deps():
    import h5py
    import numpy
    import pandas
    import torch


def test_checkpoints():
    import hashlib

    md5_book = {
        'prosit': 'ec4bc8a7761c38f8732f5f53c3ec40ff',
        'pdeep': 'ae4deb4efbc963c57bd1420ba5df109e',
    }
    for name, md5_v in md5_book.items():
        weights = models.Model_Weights_Factories(name)
        weights_bin = open(weights, 'rb')
        md5_test = hashlib.md5()
        md5_test.update(weights_bin.read())
        assert md5_test.hexdigest() == md5_v
