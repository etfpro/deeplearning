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
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    return -np.sum(t * np.log(y + 1e-7)) / batch_size



################################################################################
#
# 미분 관련 함수
#
################################################################################

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 기울기
def numerical_gradient(f, x):
    h = 1e-4

    grad = np.zeros_like(x)

    print("size of x =", x.size)
    for i in range(x.size):
        org_x = x[i]

        x[i] = org_x + h
        f1 = f(x)

        x[i] = org_x - h
        f2 = f(x)

        grad[i] = (f1 - f2) / (2 * h)

        x[i] = org_x

    return grad


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

    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
