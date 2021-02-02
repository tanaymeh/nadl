import sys
sys.path.append("/home/heytanay/Desktop/nadl/nadl/core")

import numpy as np
from tensor import Tensor

print(f"{'='*20} Running Basic Checks {'='*20}")


a = Tensor(data=np.array([[1,2,3], [2,3,4]]))
b = Tensor.ones_like(a)
c = Tensor.zeros_like(b)
d = Tensor.random_like(c)
print(a)
print(b)
print(c)
print(d)

print(f"{'='*20} Running Intermediate Checks {'='*20}")
set_value = np.array([[10, 20, 30], [40, 50, 60]])
e = Tensor(data=set_value)
f = Tensor.ones_like(e)
g = e * 1
h = e * 7
i = e + f
# for x_temp in [e, f, g, h, i]:
#     print(f"{x_temp}")
# print(e.data)
# print(f.data)
# print(h.data)
# print(i.data)

# i.backward()
# print(e.grad)
# print(f.grad)
# print(i.grad)

print(f"{'='*20} Running Advanced Checks {'='*20}")

e = Tensor(data=set_value)
eT = Tensor(data=e.data.T)
x = Tensor.random_like(eT)

print(e.data)
print(x.data)
print()
k = e.matmul(x)

k.backward()
print(e.grad)
print(x.grad)

# except Exception as e:
#     print(e)