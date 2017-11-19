import neuralNetwork as nn
import numpy as np
import scipy.misc as misc
import scipy.ndimage as ndimage


################################################################################
# 신경망 학습
################################################################################
file = open("../dataset/mnist_train_100.csv", "r")
training_data_list = file.readlines()
file.close()

inputNodes = len(training_data_list[0].split(',')[1:])
outputNodes = 10
learningRate = 0.05

network = nn.neuralNetwork(inputNodes, outputNodes, learningRate)

epochs = 3

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')

        # 입력값을 0.01 ~ 1.0으로 조정 (입력값이 0인 경우 가중치 학습이 안되기 때문에)
        inputs = network.normalizeInputs(all_values[1:], 255.0)

        # 레이블(실제값은 0.99, 나머지는 0.01)
        labels = np.zeros(outputNodes) + 0.01
        labels[int(all_values[0])] = 0.99

        network.train(inputs, labels)

        # 회전시킨 이미지 학습
        inputs_plusx_img = ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1, reshape=False)
        network.train(inputs_plusx_img.reshape(784), labels)

        inputs_minusx_img = ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1, reshape=False)
        network.train(inputs_minusx_img.reshape(784), labels)



################################################################################
# 신경망 테스트(성능 평가)
################################################################################
img_array = misc.imread("../dataset/2.png", flatten=True)
img_data = 255.0 - img_array.reshape(784)
print("Correct Label is 2")
network.forward(network.normalizeInputs(img_data, 255.0))

img_array = misc.imread("../dataset/3.png", flatten=True)
img_data = 255.0 - img_array.reshape(784)
print("Correct Label is 3")
network.forward(network.normalizeInputs(img_data, 255.0))

img_array = misc.imread("../dataset/3(2).png", flatten=True)
img_data = 255.0 - img_array.reshape(784)
print("Correct Label is 3")
network.forward(network.normalizeInputs(img_data, 255.0))

img_array = misc.imread("../dataset/4.png", flatten=True)
img_data = 255.0 - img_array.reshape(784)
print("Correct Label is 4")
network.forward(network.normalizeInputs(img_data, 255.0))

img_array = misc.imread("../dataset/5(2).png", flatten=True)
img_data = 255.0 - img_array.reshape(784)
print("Correct Label is 5")
network.forward(network.normalizeInputs(img_data, 255.0))

img_array = misc.imread("../dataset/6.png", flatten=True)
img_data = 255.0 - img_array.reshape(784)
print("Correct Label is 6")
network.forward(network.normalizeInputs(img_data, 255.0))

img_array = misc.imread("../dataset/6(2).png", flatten=True)
img_data = 255.0 - img_array.reshape(784)
print("Correct Label is 6")
network.forward(network.normalizeInputs(img_data, 255.0))


file = open("../dataset/mnist_test_10.csv", "r")
test_data_list = file.readlines()
file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    outputs = network.forward(network.normalizeInputs(all_values[1:], 255.0), False)

    correct_label = int(all_values[0])
    print(correct_label, "Correct Label")

    label = np.argmax(outputs)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = np.asarray(scorecard)
print("Scorecard =", scorecard)
print("Performance = ", 100.0 * scorecard_array.sum() / scorecard_array.size, "%")



"""
import matplotlib.pyplot as plt
image_array = np.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.xticks([]) # x축 눈금
plt.yticks([]) # y축 눈금
plt.show()
"""