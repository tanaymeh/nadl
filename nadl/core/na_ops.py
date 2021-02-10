"""
Use these operations rather than the overloaded ones.
Why? Because it is more Functional-Python-ic than 
just calling an Operator that you can use on any datatype
"""
import numpy as np
from ..core.tensor import Tensor
from ..core.ops import HiddenOps
from ..other.utils import Utils
from typing import Union

def add(*args) -> Tensor:
    """
    Add 2 Tensors or a Tensor and a Scalar.
    Use this instead of overloaded "+"
    """
    # Sanity check and convert these inputs to right format
    tensor1 = args[0]
    tensor2 = args[1]
    if not isinstance(tensor1, Tensor):
        tensor1 = Tensor(data=tensor1)
    if not isinstance(tensor2, Tensor):
        tensor2 = Tensor(data=tensor2)

    output = Tensor(data=tensor1.numpy() + tensor2.numpy(), _children=(tensor1, tensor2), _op='+')

    def _backward():
        __grad_check = Utils.checkGradDep(tensor1, tensor2)
        if not __grad_check:
            raise RuntimeError("Cannot perform backward propagation on a Static Tensor")

        tensor1.grad += output.grad
        tensor2.grad += output.grad
    output._backward = _backward

    return output

def pow(*args) -> Tensor:
    """
    Raises a Tensor to power of a Scalar
    Use this instead of overloaded "*"
    """
    raise NotImplementedError("Power function is broken, please use explicit methods. Sorry : (")
    # Sanity check and convert these inputs to right format
    tensor = args[0]
    power = args[1]
    if not isinstance(tensor, Tensor):
        tensor = Tensor(data=tensor)
    if not isinstance(power, Union[int, float]):
        raise RuntimeError("Only int/float powers are allowed.")
    
    output = Tensor(data=tensor.numpy()**power, _children=(tensor,), _op=f"^{power}")

    def _backward():
        __grad_check = Utils.checkGradDep(tensor)
        if not __grad_check: 
            raise RuntimeError("Cannot perform backward propagation on a Static Tensor")

        output.grad += (power * tensor.numpy() ** (power - 1)) * output