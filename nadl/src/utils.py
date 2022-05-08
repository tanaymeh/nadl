from typing import Type
import numba
import numpy as np


def get_device(device):
    if "cpu" in device:
        return "cpu"
    elif "gpu" in device:
        if numba.cuda.is_available():
            return "gpu"
        else:
            print("CUDA not found, defaulting to CPU")
            return "cpu"
    else:
        raise TypeError("Device must be either 'cpu' or 'gpu'")


def same_device_check(self, value):
    if self.device == value.device:
        return True
    else:
        return False
