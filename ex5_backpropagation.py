import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

w1 = np.random.rand(2, 2)
w2 = np.random.rand(2, 1)
b1 = np.random.rand(2)
b2 = np.random.rand(1)
lr = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for _ in range(10000):
    h = sigmoid(np.dot(X, w1) + b1)
    o = sigmoid(np.dot(h, w2) + b2)
    e = y - o
    o_grad = e * o * (1 - o)
    h_grad = np.dot(o_grad, w2.T) * h * (1 - h)
    w2 += np.dot(h.T, o_grad) * lr
    w1 += np.dot(X.T, h_grad) * lr
    b2 += np.sum(o_grad, axis=0) * lr
    b1 += np.sum(h_grad, axis=0) * lr

print("Predicted Output:\n", o)
