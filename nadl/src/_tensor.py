import numpy as np


class CoreTensorClass:
    def __init__(self, data: Union[float, list, np.ndarray], requires_grad: bool=True, children=List[Children]=None):
        self._data = is_array(data)
        self.requires_grad = requires_grad
        self.children = children
        self.shape = self._data.shape
        self.size = self._data.size
        self.grad: Optional['Tensor'] = None
        
        if self.requires_grad:
            self.zero_grad()
