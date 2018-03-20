import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256 # 은닉층 수
n_input = 28 * 28 # 입력층 수
n_noise = 128 # 노이즈 데이터의 크기


################################################################################
# 학습 진행
################################################################################

# MNIST 데이터 다운로드
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)



