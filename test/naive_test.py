import numpy as np
import tensorflow as tf
from ..nadl.core.tensor import Tensor

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
i = e + f
for x_temp in [e, f, i]:
    print(f"{x_temp}")
print(e.data)
print(f.data)
print(i.data)

i.backward()
print(e.grad)
print(f.grad)
print(i.grad)

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

tensor_fr1 = np.random.randn(5,5).astype('float32')
tensor_fr2 = np.random.randn(5,3).astype('float32')

def test_nadl():
    t1 = Tensor(data=tensor_fr1)
    t2 = Tensor(data=tensor_fr2)
    out = t1.matmul(t2)
    out.backward()
    return out.data, t1.grad, t2.grad

def test_tf():
    t1 = tf.Variable(tensor_fr1)
    t2 = tf.Variable(tensor_fr2)
    with tf.GradientTape(persistent=True) as gr:
        gr.watch(t1)
        gr.watch(t2)
        out = tf.matmul(t1, t2)
    t1_grad = gr.gradient(out, t1)
    t2_grad = gr.gradient(out, t2)

    return out.numpy(), t1_grad.numpy(), t2_grad.numpy()

te_out, te_g1, te_g2 = test_nadl()
tf_out, tf_g1, tf_g2 = test_tf()
print(f"{'='*20} Running Cross Platform Checks {'='*20}")
np.testing.assert_allclose(te_out, tf_out, atol=1e-5, err_msg="Outputs not in the range")
np.testing.assert_allclose(te_g1, tf_g1, atol=1e-5, err_msg="Gradients of T1 not in the range")
np.testing.assert_allclose(te_g2, tf_g2, atol=1e-5, err_msg="Gradients of T2 not in the range")
