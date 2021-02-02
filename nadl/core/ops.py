# Non-Overloaded function will be here
import numpy as np

class BasicOps:
    def matmul(tensor1, tensor2, dataType):
        """
        Hidden Matrix Multiplication functions that will be called in the main tensor.py file
        """
        try:
            output = dataType(data=np.matmul(tensor1.data, tensor2.data))
        except:
            raise RuntimeError(f"Invalid Matrix Multiplication, {tensor1.data.shape} is not compatible with {tensor2.data.shape}")

        def _backward():
            new_self_grad = np.matmul(output.grad, tensor2.data.T)
            new_tensor_grad = np.matmul(tensor1.data.T, output.grad)
            tensor1.grad = new_self_grad
            tensor2.grad = new_tensor_grad
            # np.add(tensor1.grad, new_self_grad, out=tensor1.grad, casting='unsafe')
            # np.add(tensor2.grad, new_tensor_grad, out=tensor2.grad, casting='unsafe')
        output._backward = _backward
        return output
    
    def relu(tensor1, dataType):
        check = 0 if tensor1.data < 0 else tensor1.data
        output = dataType(data=check, _children=(tensor1), _op='ReLU')

        def _backward():
            tensor1.grad += (output.data > 0) * output.grad
        output._backward = _backward

        return output