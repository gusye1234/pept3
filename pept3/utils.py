import logging
from copy import deepcopy

import pandas as pd

try:
    from rich.logging import RichHandler

    logging.basicConfig(
        format='%(message)s', datefmt='%H:%M:%S', handlers=[RichHandler()]
    )
except:
    logging.basicConfig(
        format='%(asctime)s-[%(name)8s] > %(message)s', datefmt='%H:%M:%S'
    )

LOGGING_LEVEL = logging.WARNING


def get_logger(name='default'):
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    return logger


def reshape_dims(array, MAX_SEQUENCE=30, ION_TYPES='yb', MAX_FRAG_CHARGE=3):
    n, dims = array.shape
    assert dims == 174
    return array.reshape(
        *[array.shape[0], MAX_SEQUENCE - 1, len(ION_TYPES), MAX_FRAG_CHARGE]
    )


def mask_outofrange(array, lengths, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        array[i, lengths[i] - 1 :, :, :] = mask
    return array


def mask_outofcharge(array, charges, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        if charges[i] < 3:
            array[i, :, :, charges[i] :] = mask
    return array


def create_mask(peptides, pad_id=0):
    return peptides == pad_id


def set_seed(seed):
    import random

    import numpy as np
    import pandas as pd
    import torch

    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """

    from time import time

    NAMED_TAPE = {}

    @staticmethod
    def dict(select_keys=None):
        if select_keys is None:
            return deepcopy(timer.NAMED_TAPE)
        else:
            new_key = {}
            for key in select_keys:
                new_key[key] = timer.NAMED_TAPE[key]
            return new_key

    @staticmethod
    def zero(select_keys=None, save_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                if (save_keys is not None) and (key in save_keys):
                    continue
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, name, **kwargs):
        if not timer.NAMED_TAPE.get(name):
            timer.NAMED_TAPE[name] = 0.0
        self.named = name
        if kwargs.get('group'):
            # TODO: add group function
            pass

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        timer.NAMED_TAPE[self.named] += timer.time() - self.start


def download_file(url, local_path):
    from urllib import request

    result = request.urlretrieve(url, local_path)
    return result
