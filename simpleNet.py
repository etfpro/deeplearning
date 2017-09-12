#simpleNet.py


import numpy as np
import functions as func


# 단층 신경망
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # [2, 3] 가중치를 표준정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = func.softmax(z)
        return func.cross_entropy_error(y, t)


net = simpleNet();

x = np.array([0.6, 0.9])
y = net.predict(x)
print(y)

t = np.array([0, 0, 1])
print(net.loss(x, t))

