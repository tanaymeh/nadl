# Naive Automatic Differentiation Library - NADL

![NADL Logo](assets/NADL.gif)


[![Build Status](https://travis-ci.com/heytanay/nadl.svg?branch=main)](https://travis-ci.com/heytanay/nadl)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/heytanay/nadl/blob/main/LICENSE)

A little Automatic Differentiation (**read: DEEP LEARNING**) library that I wrote in Python using only numpy (for doing linear algebra with ease).

This framework can do everything (as long as calculating gradients is concerned) that any other big framework out there can do.

## Installation
I am currently adding more functionality to the core functions of this framework and this is why the installation for the time being is a little complex but I'll figure it out when I have cleared the back-log of existing requests.

To use this framework, you currently only need to import `tensor.py` file and it will all work just fine. All other modules are imported into `tensor.py` so you are covered.

### Hardware Support
I am currently contemplating the idea of using custom hardware like GPU so fasten the calculations and processing using [`pyopencl`](https://documen.tician.de/pyopencl/) but so far it's just on the white-board (yes, I use white boards).

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
I have added a few basic unit tests that use Tensorflow's `tf.GradientTape()` to check it the gradients being calculated by my code and those calculated by tensorflow are in the same range.

Note: They cannnot be exactly same (though they are precise to 4 decimal digits) since Tensorflow is more efficient as they do all the heavy lifting using a C-backend so this is the best precision you will get from **nadl**.

To run the test, execute the following:

```shell
$ python -m pytest
```
The above command should be executed in the repository's root directory.

## Conclusion and Collaboration

I am planning on to adding more features to this little frameworks as well as optimizations and hard-coded algorithms for faster runtime.

I have written all this by myself and so it's hard to push out changes fast enough or to implement new functionalities.

If you like this idea and want to collaborate, please contact me on [LinkedIn](https://www.linkedin.com/in/tanaymehta28/) or by [E-Mail](mailto:heyytanay@gmail.com). I will be more than happy to collab!

<hr>
Made with 🖤 by Tanay Mehta

Inspired by [Andrej Karpathy's Micrograd](https://github.com/karpathy/micrograd).
