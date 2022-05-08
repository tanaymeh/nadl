import numpy as np


class CPUOps:
    def __init__(self):
        pass

    def add(self, tensor1, tensor2):
        data = tensor1.data + tensor2.data
        requires_grad = tensor1.requires_grad or tensor2.requires_grad

        pass
