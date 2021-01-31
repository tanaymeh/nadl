import sys
sys.path.append("/home/heytanay/Desktop/nadl/nadl/core")

import numpy as np
from tensor import Tensor

print(f"{'='*20} Running Basic Checks {'='*20}")

try:
    a = Tensor(data=np.array([[1,2,3], [2,3,4]]))
    b = Tensor.ones_like(a)
    c = Tensor.zeros_like(b)
    d = Tensor.random_like(c)
    print(a)
    print(b)
    print(c)
    print(d)

except Exception as e:
    print(e)