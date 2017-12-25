#simpleNet.py


import numpy as np
import functions as func


# 단층 신경망
class simpleNet:
    def __init__(self):
        # 가중치 초기화: 표준정규분포
        self.W = np.random.randn(2, 3)


    def predict(self, x):
        return np.dot(x, self.W)


    # 손실함수
    # - 신경망의 전체 추론과정을 거쳐서 출력값을 계산하여 정답과 비교하는 방식
    # - 손실함수 계산이 오래걸린다.
    # - 중간 계산 결과를 저장하면 좀 더 효율적인 계산이 가능
    def loss(self, x, t):
        z = self.predict(x)
        y = func.softmax(z)
        loss = func.cross_entropy_error(y, t)
        return loss



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
