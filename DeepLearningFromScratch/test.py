from mnist import load_mnist
import numpy as np
from twoLayerNet import TwoLayerNet


# 데이터 읽기
(x_train, t_train), _ = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 3개의 훈련 데이터
x_batch = x_train[:3]
t_batch = t_train[:3]

# 3개의 훈련 데이터에 대해 미분 실시
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
for key in grad_numerical.keys():
    d = np.abs(grad_backprop[key] - grad_numerical[key])
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
