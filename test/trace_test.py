import os
import sys
import numpy as np

sys.path.append("../")
from nadl.core.tensor import Tensor

tensor_fr1 = np.random.randn(5,5).astype('float32')
tensor_fr2 = np.random.randn(5,3).astype('float32')

t1 = Tensor(data=tensor_fr1)
t2 = Tensor(data=tensor_fr2)

out = t1.matmul(t2)
out.backward()

# print(t1.data, t2.data)