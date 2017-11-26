#simpleNet.py


import numpy as np
import functions as func


# 단층 신경망
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # [2, 3] 가중치를 표준정규분포로 초기화

    # forward
    def predict(self, x):
        return np.dot(x, self.W)

    # 손실함수
    def loss(self, x, t):
        z = self.predict(x)
        y = func.softmax(z)
        return func.cross_entropy_error(y, t)




net = simpleNet()
print(">> Initial W <<\n", net.W)

x = np.array([0.6, 0.9])
y = net.predict(x)
print(">> y <<\n", y)
print(np.argmax(y))

t = np.array([0, 0, 1])
loss = net.loss(x, t)
print(">> loss <<\n", loss)


dW = func.numerical_gradient(lambda w: net.loss(x, t), net.W)
print(">> delta W <<\n", dW)
