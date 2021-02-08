import sys
sys.path.append("../")
import numpy as np
import sklearn.datasets as ds
from nadl.core.tensor import Tensor

# Get the data
X, y = ds.load_boston(return_X_y=True)
X = Tensor(data=X)
y = Tensor(data=y)

# Init weights
theta = np.random.uniform(size=(X.shape[1], 1))
theta = Tensor(theta)

# Some helper functions
def predict():
    # X(?, 13) * theta(13, 1) = y_hat(?, 1)
    return X.matmul(theta)

def calc_loss(y: Tensor, y_hat: Tensor):
    loss = y_hat - y
    loss = loss * 2
    loss = Tensor(
        data=np.sum(loss.numpy)
    )
    return loss