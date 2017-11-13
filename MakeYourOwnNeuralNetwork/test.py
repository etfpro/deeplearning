import neuralNetwork as nn
import numpy as np
import matplotlib.pyplot as plt


file = open("../dataset/mnist_train_100.csv")
training_data_list = file.readlines()
file.close()

inputNodes = len(training_data_list[0].split(',')[1:])
outputNodes = 10
hiddenNodes = 100
learningRate = 0.3

network = nn.neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

# 신경망 학습
for record in training_data_list:
    all_values = record.split(',')
    # 입력값을 0.01 ~ 1.0으로 조정 (입력값이 0인 경우 가중치 학습이 안되기 때문에)
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # 레이블(실제값은 0.99, 나머지는 0.01)
    labels = np.zeros(outputNodes) + 0.01
    labels[int(all_values[0])] = 0.99

    network.train(inputs, labels)



# 신경망 테스트
file = open("../dataset/mnist_test_10.csv")
test_data_list = file.readlines()
file.close()

all_values = test_data_list[0].split(',')
result = network.query((np.asfarray(all_values[1:])))
print("result = %d[%f]" % (np.argmax(result), np.max(result)))



"""
image_array = np.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.xticks([]) # x축 눈금
plt.yticks([]) # y축 눈금
plt.show()
"""