# Layers.py
# 역전파 시 해당 함수를 국소적 미분한 값을 곱한다.

import numpy as np


# Affine 변환 계층 (X·W + B)
class Affine:
    pass



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


if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])



