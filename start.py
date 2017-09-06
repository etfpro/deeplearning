#start.py

import numpy as np
import matplotlib.pylab as plt

def identity(x):
    return x

def step_func(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x)) # 배열 원소 중 가장 큰 수를 빼서 overflow 방지
    return exp_x / np.sum(exp_x)


def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network;

def forward(network, x):
    a1 = np.dot(x, network['w1']) + network['b1']
    z1 = sigmoid(a1)

    a2 = np.dot(z1, network['w2']) + network['b2']
    z2 = sigmoid(a2)

    a3 = np.dot(z2, network['w3']) + network['b3']
    #y = softmax(a3)
    y = identity(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)

print(y)
