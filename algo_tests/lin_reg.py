import sys
sys.path.append("../")
import numpy as np
import sklearn.datasets as ds
from nadl.core.tensor import Tensor 

def run_training():
    # Get the data
    X, y = ds.load_boston(return_X_y=True)
    X = Tensor(data=X)
    y = Tensor(data=y)

    # Init weights
    theta = np.random.uniform(size=(X.shape[1], 1))
    theta = Tensor(theta)

    # Run for 100 epochs
    for i in range(100):
        preds = X.matmul(theta)
        loss = (y-preds)*2
        loss.backward()
        temp_theta = theta.grad * 0.001
        theta = theta - Tensor(temp_theta)
        print(preds.grad)
        # print(cost.numpy)
    print(type(temp_theta))
    # print(X.grad)
    # print(y.grad)

if __name__ == "__main__":
    run_training()