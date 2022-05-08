import numpy as np
from src._tensor import CoreTensorClass
from typing import List, NamedTuple, Callable, Optional, Union

class Children(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]

def is_tensor(tensor):
    if isinstance(tensor, 'Tensor'):
        return tensor
    else:
        return Tensor(tensor)
    
def is_array(array):
    if isinstance(array, Union[float, list, np.ndarray]):
        return array
    else:
        return np.array(array)
    
class Tensor(CoreTensorClass):
    def __init__(self, data: Union[float, list, np.ndarray], requires_grad: bool=True, children=List[Children]=None):
        self._data = is_array(data)
        self.requires_grad = requires_grad
        self.children = children
        self.shape = self._data.shape
        self.size = self._data.size
        self.grad: Optional['Tensor'] = None
        
        if self.requires_grad:
            self.zero_grad()
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        self.grad = None
    
    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype
    
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))
    
    def __repr__(self) -> str:
        return "Tensor(data={}, dtype={}, requires_grad={})".format(self.data, self.dtype, self.requires_grad)