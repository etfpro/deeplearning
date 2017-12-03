# Layers.py
# 역전파 시 해당 함수를 국소적 미분한 값을 곱한다.

import numpy as np
import functions as func

# Affine 변환 계층 (x·W + b)
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        # 순전파 입력을 역전파(미분) 시 사용을 위해 저장
        self.x = None

        self.original_x_shape = None


    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1) # 행: 데이터의 수, 열: 1개의 입력값을 1행으로 풀어놓은 상태

        out = np.dot(self.x, self.W) + self.b
        return out


    # Affine 함수 미분
    # dout은 역전파 시킬 값(이전 계층(ReLU등 활성화 함수 계층)의 미분 값)
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)

        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx



# ReLU 활성화 함수 계층
class Relu:
    def __init__(self):
        # 입력값이 0 이하인지를 나타내는 배열
        # 추후 역전파를 위한 미분에 사용
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy() # 입력 값을 변화시키지 않기 위해 복사
        out[self.mask] = 0 # x <= 0은 0, x > 0은 x
        return out

    # ReLU 함수 미분
    # dout은 역전파 시킬 값(이전 계층의 미분 값)
    def backward(self, dout):
        dout[self.mask] = 0 # ReLU 미분: 0보다 작거나 같은 경우 0, 0보다 큰 경우 1
        dx = dout
        return dx



# Sigmoid 활성화 함수 계층
class Sigmoid:
    def __init__(self):
        self.out = None


    def forwward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        out = self.out
        return out


    # Sigmoid 함수 미분
    # dout은 역전파 시킬 값(이전 계층의 미분 값)
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx;



# Softmax 활성화 함수 + Cross Entropy Error 함수 계층
# 분류 신경망의 마지막 계층으로 학습시에만 사용된다.
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실
        self.y = None # Softamx 출력
        self.t = None # 정답 레이블(One-hot Encoding

    # x: 입력
    # t: 정답 레이블(마지막 계층이기 때문에 정답 레이블이 필요)
    def forward(self, x, t):
        self.t = t
        self.y = func.softmax(x)
        self.loss = func.cross_entropy_error(self.y, self.t)
        return self.loss


    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        # 배치의 수로 나눠서 데이터 1개당 오차를 전파
        dx = (self.y - self.t) / batch_size
        return dx



if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])



