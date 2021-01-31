import numpy as np
from functools import partialmethod

class TensorDatatype:
    int32 = np.int32
    float32 = np.float32
    float64 = np.float64

class Tensor:
    def __init__(self, data: np.ndarray, dtype:TensorDatatype.float32, requires_grad: bool=True,_children: tuple=()):
        self.data = data
        self.dtype = dtype
        # self.shape = self.data.shape
        self.grad = 0
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._backward = lambda: None

        # Check if the data IS a numpy array
        try:
            # Store any other numpy specific operations here.
            isinstance(self.data, np.ndarray)
            # self.shape = self.data.shape
        except:
            raise ValueError(f"The given data: {self.data} is not a numpy.ndarray")

    # @property
    # def dtype(self):
    #     return self.dtype

    @property
    def shape(self):
        return self.shape

    @shape.setter
    def shape(self, _newShape: tuple):
        try:
            self.data = self.data.reshape(_newShape)
        except:
            raise ValueError(f"Tensor cannot be reshaped from {self.data.shape} to {_newShape}")

    # Explicitly change the datatype of a Tensor (not recommended!)
    # @dtype.setter
    # def dtype(self, datatype):
    #     if not datatype in list(vars(TensorDatatype).keys()):
    #         raise TypeError(f"Cannot change the existing data type: {self.dtype} to {datatype}")
    #     self.dtype = datatype
            
    
    def __repr__(self):
        return f"Tensor Instance with (value: {self.data.dtype}, shape: {self.data.shape})"