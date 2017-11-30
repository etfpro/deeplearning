# Layers.py
# 역전파 시 해당 함수를 국소적 미분한 값을 곱한다.

import numpy as np


# ReLU 활성화 함수
class Relu:
    def __init__(self):
        # 입력값이 0 이하인지를 나타내는 배열
        # 추후 역전파를 위한 미분에 사용
        self.mask = None

    # x는 numpy 배열로 shape은 해당 계층의 수와 동일
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy() # 입력 값을 변화시키지 않기 위해 복사
        out[self.mask] = 0 # x <= 0은 0, x > 0은 x
        return out


    # dout은 역전파 시킬 값
    def backward(self, dout):
        dout[self.mask] = 0 # ReLU 미분: 0보다 작거나 같은 경우 0, 0보다 큰 경우 1
        return dout


if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])



