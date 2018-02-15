


def halfFloatNet():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

    network = DeepConvNet()
    network.load_params("deep_convnet_params.pkl")

    sampled = 10000 # 고속화를 위한 표본추출
    x_test = x_test[:sampled]
    t_test = t_test[:sampled]

    print("caluculate accuracy (float64) ... ")
    print(network.accuracy(x_test, t_test))

    # float16(반정밀도)로 형변환
    x_test = x_test.astype(np.float16)
    for param in network.params.values():
        param[...] = param.astype(np.float16)

    print("caluculate accuracy (float16) ... ")
    print(network.accuracy(x_test, t_test))
