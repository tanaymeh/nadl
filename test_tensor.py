import numpy as np
from numpy.testing._private.utils import check_free_memory
from tensor import Tensor, TensorDatatype

print(f"{'='*20} Running Basic Checks {'='*20}")

try:
    a = Tensor(np.array([1.,2.,3.]), dtype=TensorDatatype.float64)
    b = Tensor(np.array([5.,4.,7.]), dtype=TensorDatatype.float32)
    c = Tensor(np.array([18.,26.,12.]), dtype=TensorDatatype.int32)

    print(a)
    print(b)
    print(c)

except Exception as e:
    print(e)