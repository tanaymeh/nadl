<h1 align='center'> Naive Automatic Differentiation Library - NADL</h1>

<img src='assets/NADL.gif' align='center' alt='NADL Logo'/>

<a href='https://travis-ci.com/heytanay/nadl'><img align='center' src='https://travis-ci.com/heytanay/nadl.svg?branch=main'/></a>
<a href='https://www.python.org/'><img align='center' src='https://img.shields.io/badge/Made%20with-Python-1f425f.svg'/></a>
<a href='https://github.com/heytanay/nadl/blob/main/LICENSE'><img align='center' src='https://img.shields.io/github/license/Naereen/StrapDown.js.svg'/></a>


A little Automatic Differentiation library that I wrote in Python using only numpy that can calculate first order gradients using AD.

This framework is very simple implementation of how other big framworks do gradient calculation using Numerical Automatic Differentiation.

## Installation

I am currently adding more functionality to the core functions of this framework and this is why the installation for the time being is a little complex but I'll figure it out when I have cleared the back-log of existing tasks.

To use this framework, you currently only need to import `tensor.py` file and it will all work just fine. All other modules are imported into `tensor.py` so you are covered.

### Hardware Support

I am currently contemplating the idea of using custom hardware like GPU to fasten the calculations and processing using [`pyopencl`](https://documen.tician.de/pyopencl/) but so far it's just on the white-board.

## Getting Started

The quickest way to start using **nadl**:

* Download this repository (or use Github codespaces, your wish) and open it in a terminal/file manager.
* Make a python file and import the `tensor.py` file.
* Write your code!

A nice example will be:

```python
import numpy as np
from nadl.core.tensor import Tensor

# Initialize a random numpy array and define a tensor using it
numpy_init1 = np.random.randn(5,5).astype('float32')
a = Tensor(data=nump_init1, requires_grad=True)

# Repeat for another tensor with slightly different size
numpy_init2 = np.random.randn(5,3).astype('float32')
b = Tensor(data=numpy_init2, requires_grad=True)

# Do some Operations on them
c = a.matmul(b)

# Do Backward Propagation to calculate the gradients
c.backward()

# Print out the gradients
print(a.grad)   # In leibnitz notation: dc/da
print(b.grad)   # In leibnitz notation: dc/db
```

## Testing

I have added a few basic unit tests that use Tensorflow's `tf.GradientTape()` to check if the gradients being calculated by my code and those calculated by tensorflow are in the same range.

Note: They cannnot be exactly same (though they are precise to 4 decimal digits) since Tensorflow is more efficient as they do all the heavy lifting using a C-backend so this is the best precision you will get from **nadl**.

To run the test, execute the following:

```shell
$ python -m pytest
```

The above command should be executed in the repository's root directory.

## Conclusion and Collaboration

I am planning to add more features to this little framework as well as optimizations and hard-coded algorithms for faster runtime.

I have written all this by myself (with a lot of help from from Joel Gru's Autograd and Andrey Karpathy's Micrograd) and so it's hard to push out changes fast enough or to implement new functionalities.

If you like this idea and want to collaborate, please contact me on [LinkedIn](https://www.linkedin.com/in/tanaymehta28/) or by [E-Mail](mailto:heyytanay@gmail.com). I will be more than happy to collab!

<hr>

Inspired by [Andrej Karpathy's Micrograd](https://github.com/karpathy/micrograd).

Made with 🖤 by Tanay Mehta

