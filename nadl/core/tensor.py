import numpy as np
from ..core.ops import HiddenOps
from ..other.utils import Utils

class Tensor:
    """
    Core Tensor Class
    This class will be responsible for all the heavy lifting.
    """
    def __init__(self, data: np.ndarray, requires_grad: bool=True, _children: tuple=(), _op: str=''):
        if not isinstance(data, np.ndarray):
            raise TypeError("Only Numpy arrays are supported for now.")
        
        self.data = data
        self.requires_grad = requires_grad

        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op

    def experimental_set_data(self, new_data):
        """
        Explicitly change the data of an already initialized Tensor.
        This will reset all the gradients back to 0
        """
        self.data = new_data
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        """
        Returns the shape of the Tensor
        """
        return self.data.shape
    
    @property
    def dtype(self):
        """
        Returns the data of the Tensor
        """
        return self.data.dtype
    
    @property
    def numpy(self):
        """
        Returns the data of the tensor in a numpy array.
        Use this method to retrieve the data
        """
        return np.array(self.data)

    @classmethod
    def ones_like(cls, tensor):
        """
        Returns a Tensor full of Ones with the same shape as the provided tensor
        
        Just like np.ones_like(...) function
        """
        return cls(data=np.ones(shape=tensor.shape, dtype=tensor.dtype))

    @classmethod
    def zeros_like(cls, tensor):
        """
        Returns a Tensor full of Zeros with the same shape as the provided tensor
        
        Just like np.zeros_like(...) function
        """
        return cls(data=np.zeros(shape=tensor.shape, dtype=tensor.dtype))
    
    @classmethod
    def random_like(cls, tensor):
        """
        Returns a Tensor full of Random Numbers with the same shape as the provided tensor
        
        Just like np.ones_like(...) function but instead of ones, it generates random numbers
        """
        return cls(data=np.random.rand(*tensor.shape))

    def __repr__(self):
        """
        Returns the string representation of a Tensor
        """
        return f"Tensor<shape={self.shape}, dtype={self.dtype}>"

    def __add__(self, tensor):
        """
        Overloaded function for "add" operator.
        Use na_ops.add() instead
        """
        if not isinstance(tensor, Tensor):
            tensor = Tensor(data=tensor)
        
        output = Tensor(data=self.data+tensor.data, _children=(self, tensor), _op='+')

        def _backward():
            __grad_check = Utils.checkGradDep(self, tensor)
            if not __grad_check: raise RuntimeError("Cannot perform backward propagation on a Static Tensor")

            self.grad += output.grad
            tensor.grad += output.grad
        output._backward = _backward

        return output

    def __mul__(self, scalar):
        """
        Multiplication using the " * " operator is only supported for a Tensor and a scalar value
        To multiply a Tensor with a Tensor, use the "tensor1.dot(tensor2)" method.

        Use na_ops.smul()
        """
        assert isinstance(scalar, (int, float, bool)), "Only multiplication with a scalar value is supported using '*' operator.\nFor Multiplication with a vector, use the '.dot()' function."
        
        output = Tensor(data=self.data * scalar, _children=(self,), _op='*')
        return output
    
    def __neg__(self):
        """
        Multiplies every element in the Tensor with -1
        More fancy term: "Inverts all values in a Tensor"
        """
        return self * -1

    def __div__(self, tensor):
        raise NotImplementedError("Division Operation is currently not implemented.")

    def __pow__(self, scalar):
        """
        Only raise to scalar powers
        """
        output = HiddenOps.power(tensor=self, power=scalar, TensorDataTypeWrapper=Tensor)
        return output
    
    def __sub__(self, tensor):
        """
        Subtraction between 2 tensors
        """
        output = HiddenOps.subtract(tensor1=self, tensor2=tensor, TensorDataTypeWrapper=Tensor)
        return output
    
    def relu(self):
        """
        Upper-level abstraction for ReLU

        Use na_ops.activations.relu() instead
        """
        output = HiddenOps.relu(tensor1=self, TensorDataTypeWrapper=Tensor)
        return output

    def matmul(self, tensor):
        """
        Upper-level abstraction for the matrix multiplication function

        Use na_ops.matmul() instead
        """
        output = HiddenOps.matmul(tensor1=self, tensor2=tensor, TensorDataTypeWrapper=Tensor)
        return output
    
    def sum(self):
        """
        Upper-level abstraction for Tensor sum function
        """
        output = HiddenOps.tensor_sum(tensor=self, TensorDataTypeWrapper=Tensor)
        return output

    def backward(self):
        """
        This function will perform the backward propagation

        Recursively visit all the nodes in the graph and then call the backward function on
        the nodes.
        
        Topological Sort.
        """
        topology = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(v)
                topology.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topology):
            v._backward()