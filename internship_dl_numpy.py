import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm


def main():

    N, D_in, D_out, H = 100, 1000, 10, 64
    batch = 100

    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    w1 = np.random.randn(D_in, H)
    b1 = np.random.randn(N, H)
    w2 = np.random.randn(H, D_out)
    b2 = np.random.randn(N, D_out)

    losses = np.zeros(batch)
    lr = 1e-6

    for i in tqdm(range(batch)):
        # forward
        h = x.dot(w1) + b1
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2) + b2

        # loss
        losses[i] = np.square(y_pred - y).mean()
        print("{}: {}".format(i, losses[i]))

        # backward
        grad_y_pred = 2 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_b2 = grad_y_pred
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)
        grad_b1 = grad_h

        # update
        w2 = w2 - lr * grad_w2
        b2 = b2 - lr * grad_b2
        w1 = w1 - lr * grad_w1
        b1 = b1 - lr * grad_b1

    plt.figure(1)
    plt.plot(losses)
    plt.show()
    print(y[:1][:])
    print(y_pred[:1][:])



if __name__ == "__main__":
    main()


