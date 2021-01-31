import numpy as np
from functools import partialmethod

class Tensor:
    """
    Core Tensor Class.
    This is the building block of this mini-framework.
    """
    def __init__(self, data: np.ndarray, requires_grad: bool=True, _children: tuple=(), _op: str=''):
        if not isinstance(data, np.ndarray):
            raise TypeError("Only Numpy arrays are supported for now.")
        
        self.data = data
        self.requires_grad = requires_grad

        self.grad = None
        self._prev = set(_children)
        # self._op = _op

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @classmethod
    def ones_like(cls, tensor):
        """
        Returns a Tensor full of Ones with the same shape as the provided tensor
        """
        return cls(data=np.ones(shape=tensor.shape, dtype=tensor.dtype))

    @classmethod
    def zeros_like(cls, tensor):
        """
        Returns a Tensor full of Zeros with the same shape as the provided tensor
        """
        return cls(data=np.zeros(shape=tensor.shape, dtype=tensor.dtype))
    
    @classmethod
    def random_like(cls, tensor):
        """
        Returns a Tensor full of Random Numbers with the same shape as the provided tensor
        """
        return cls(data=np.random.random(size=tensor.shape))

    def __repr__(self):
        return f"Tensor<shape={self.shape}, dtype={self.dtype}>"

    def __get_data(self):
        """
        Experimental function - only for internal workings.
        """
        return self.data