import numpy as np
from functools import partialmethod
from numpy.lib.arraysetops import isin

from numpy.lib.function_base import gradient

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

        self.grad = 0
        self._prev = set(_children)
        self._op = _op

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
    
    def __add__(self, tensor):
        if not isinstance(tensor, Tensor):
            tensor = Tensor(data=tensor)
        
        output = Tensor(data=self.data+tensor.data, _children=(self, tensor), _op='+')

        def _backward():
            self.grad += output.grad
            tensor.grad += output.grad
        output._backward = _backward

        return output

    def __mul__(self, scalar):
        """
        Multiplication using the " * " operator is only supported for a Tensor and a scalar value
        To multiply a Tensor with a Tensor, use the "tensor1.dot(tensor2)" method.
        """
        assert isinstance(scalar, (int, float, bool)), "Only multiplication with a scalar value is supported using '*' operator.\nFor Multiplication with a vector, use the '.dot()' function."
        
        output = Tensor(data=self.data * scalar, _children=(self,), _op='*')
        return output
    
    def __div__(self, tensor):
        raise NotImplementedError("Division Operation is currently not implemented.")

    def __pow__(self, scalar):
        """
        Only raise to scalar powers
        """
        assert isin(scalar, (int, float)), "Only int/float powers are allowed."

        output = Tensor(self.data ** scalar, _children=(self,), _op=f"^{scalar}")

        def _backward():
            self.grad += (scalar * self.data ** (scalar - 1)) * output
        output._backward = _backward

        return output
    
    def relu(self):
        __check = 0 if self.data < 0 else self.data
        output = Tensor(data=__check, _children=(self), _op='ReLU')

        def _backward():
            self.grad += (output.data > 0) * output.grad
        output._backward = _backward

        return output