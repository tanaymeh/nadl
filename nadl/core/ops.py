# Non-Overloaded function will be here
import numpy as np
from ..other.utils import Utils

class HiddenOps:
    def matmul(tensor1, tensor2, TensorDataTypeWrapper):
        """
        Hidden Matrix Multiplication function that will be called in the main tensor.py file
        """
        try:
            output = TensorDataTypeWrapper(data=np.matmul(tensor1.data, tensor2.data))
        except:
            raise RuntimeError(f"Invalid Matrix Multiplication, {tensor1.data.shape} is not compatible with {tensor2.data.shape}")

        def _backward():
            __grad_check = Utils.checkGradDep(tensor1, tensor2)
            if not __grad_check: raise RuntimeError("Cannot perform backward propagation on a Static Tensor")
            new_self_grad = np.matmul(output.grad, tensor2.data.T)
            new_tensor_grad = np.matmul(tensor1.data.T, output.grad)
            tensor1.grad = new_self_grad
            tensor2.grad = new_tensor_grad
            # np.add(tensor1.grad, new_self_grad, out=tensor1.grad, casting='unsafe')
            # np.add(tensor2.grad, new_tensor_grad, out=tensor2.grad, casting='unsafe')
        output._backward = _backward
        return output
    
    def relu(tensor1, TensorDataTypeWrapper):
        """
        Internal relu function code that will be invoked when calling on a tensor object
        """
        check = 0 if tensor1.data < 0 else tensor1.data
        output = TensorDataTypeWrapper(data=check, _children=(tensor1), _op='ReLU')

        def _backward():
            __grad_check = Utils.checkGradDep(tensor1)
            if not __grad_check: raise RuntimeError("Cannot perform backward propagation on a Static Tensor")
    
            tensor1.grad += (output.data > 0) * output.grad
        output._backward = _backward

        return output
    
    def tensor_sum(tensor, TensorDataTypeWrapper):
        """
        Returns the sum of all elements of a Tensor in a new Tensor
        """
        output = TensorDataTypeWrapper(data=tensor.numpy.sum())

        def _backward():
            __grad_check = Utils.checkGradDep(tensor)
            if not __grad_check: raise RuntimeError("Cannot perform backward propagation on a Static Tensor")
            
            tensor.grad = tensor.grad * np.ones_like(tensor.numpy)
        output._backward = _backward

        return output