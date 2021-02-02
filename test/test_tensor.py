import sys
sys.path.append("/home/heytanay/Desktop/nadl/nadl/core")

from tensor import Tensor
import unittest
import numpy as np
import tensorflow as tf

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



class TestNADL(unittest.TestCase):
    def tensor_tests():    
        te_out, te_g1, te_g2 = test_nadl()
        tf_out, tf_g1, tf_g2 = test_tf()

        np.testing.assert_allclose(te_out, tf_out, atol=1e-5, err_msg="Outputs not in the range")
        np.testing.assert_allclose(te_g1, tf_g1, atol=1e-5, err_msg="Gradients of T1 not in the range")
        np.testing.assert_allclose(te_g2, tf_g2, atol=1e-5, err_msg="Gradients of T2 not in the range")

if __name__ == "__main__":
    unittest.main()