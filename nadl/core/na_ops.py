"""
Use these operations rather than the overloaded ones.
Why? Because it is more Functional-Python-ic than 
just calling an Operator that you can use on any datatype
"""
import numpy as np
from ..other.utils import Utils

def add(tensor1, tensor2, dataType):
    """
    Add 2 Tensors or a Tensor and a Scalar.
    Use this instead of overloaded "+"
    """
    # Sanity check and convert these inputs to right format
    if not isinstance(tensor1, dataType):
        tensor1 = dataType(data=tensor1)
    if not isinstance(tensor2, dataType):
        tensor2 = dataType(data=tensor2)

    output = dataType(data=tensor1.numpy() + tensor2.numpy(), _children=(tensor1, tensor2), _op='+')

    def _backward():
        __grad_check = Utils.checkGradDep(tensor1, tensor2)
        if not __grad_check:
            raise RuntimeError("Cannot perform backward propagation on a Static Tensor")

        tensor1.grad += output.grad
        tensor2.grad += output.grad
    output._backward = _backward

    return output