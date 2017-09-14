#twoLayerNet.py

import numpy as np
import functions as func


class TwoLayerNet:
    def __init(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}

        # hidden layer 가중치, 편차 초기화
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeors(hidden_size)

        # output layer 가중치, 편차 초기화
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # hidden layer 출력값
        z1 = func.sigmoid(np.dot(x, W1) + b1)

        # output layer 출력값
        y = func.softmax(np.dot(z1, W2) + b2)

        return y


    def loss(self, x, t):
        y = self.predict(x)
        return func.cross_entropy_error(y, t)


    def accuracy(self, x, t):
        return np.sum(np.argmax(self.predict(x), axis=1) == np.argmax(t, axis=1)) / float(x.shape[0])


    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}

