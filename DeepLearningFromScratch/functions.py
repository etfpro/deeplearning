#start.py

import numpy as np
import matplotlib.pylab as plt
import os


################################################################################
#
# Activation Functions (활성화 함수들)
#
################################################################################

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



class Relu:
    def __init__(self):
        self.mask = None


    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def forward2(self, x):
        self.mask = (x <= 0)
        return np.max(0, x)


    def backward(self, dout):
        dout[self.mask] = 0
        return dout;



class Sigmoid:
    def __init__(self):
        self.out = None;


    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-p))
        return self.out


    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out








################################################################################
#
# Activation Functions (활성화 함수들)
#
################################################################################

# 평균제곱오차 함수
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


# 교차엔트로피오차 함수
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size



################################################################################
#
# 미분 관련 함수
#
################################################################################

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 기울기 계산 (미분)
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad


# 경사하강 기울기 조절
def gradient_descent(f, init_w, lr=0.01, step_num=100):
    w = init_w
    w_history = []

    for i in range(step_num):
        w_history.append(w.copy())
        w -= lr * numerical_gradient(f, w)

    return w, np.array(w_history)










################################################################################
#
# 테스트
#
################################################################################

def function_1(x):
    return 0.01 * x**2 + 0.1 * x

def function_2(x):
    return np.sum(x**2)



if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    print(x)
