# coding: utf-8
import urllib.request
import os.path
import gzip
import pickle
import os
import numpy as np
import functions as func


url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "/../dataset"
pickle_path = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)


# mnist 레이블 파일을 numpy 배열로 읽는다.
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    # mnist 레이블 파일의 8바이트 이후 부터 읽는다(첫 8바이트는 ?)
    # 1바이트(8비트) 정수배열로 읽음
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels


# MNIST 이미지 파일을 numpy 배열(데이터수 X 784)로 읽는다.
def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    # mnist 이미지 파일의 16바이트 이후 부터 읽는다(첫 16바이트는 ?)
    # 1바이트(8비트) 정수배열로 읽음
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    print("Done")
    
    return data


# MNIST 파일들을 읽어서 numpy 배열로 변환
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset


# MNIST 데이터를 다운받아서 numpy 배열로 변환하여 pickle 파일로 저장
def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    with open(pickle_path, 'wb') as f:
        pickle.dump(dataset, f, -1)


# 레이블 데이터를 10개의 출력으로 one-hot encoding (입력데이터 수 X 10)
def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

# MNIST 데이터셋 읽어오기
# Returns: (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
def load_mnist(normalize=True, flatten=True, one_hot_label=False):

    # MNIST 데이터 셋을 다운받아서 읽어들여 numpy 배열로 변환하여 pickle 파일로 저장
    if not os.path.exists(pickle_path):
        init_mnist()

    # MNIST 데이터 셋 pickle 피일을 읽는다.
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)

    # 입력값을 0.0 ~ 1.0 사이의 실수값으로 정규화
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # 레이블에 대한 one-hot encoding
    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
            #dataset[key] = dataset[key].reshape(-1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


################################################################################

# 테스트 데이터와 레이블 로드
def getTestData():
    _, (test_data, test_label) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return test_data, test_label


def initNetwork():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    # 1 레이어
    a1 = np.dot(x, W1) + b1
    z1 = func.sigmoid(a1)

    # 2 레이어
    a2 = np.dot(z1, W2) + b2
    z2 = func.sigmoid(a2)

    # 3 레이어(출력)
    a3 = np.dot(z2, W3) + b3
    y = func.softmax(a3)

    return y



################################################################################

def predictTest():
    # MNIST 테스트용 데이터(10,000개) 로드
    test_data, test_label = getTestData()

    dataCount = len(test_label)
    print("Count of test data: ", dataCount)

    batch_size = 100

    accuracy_cnt = 0

    # 미리 학습된 신경망 로드
    network = loadTrainedNetwork()
    for i in range(0, dataCount, batch_size):
        x_batch = test_data[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == test_label[i:i+batch_size])

    print("Accuracy: %.2f%%" % (accuracy_cnt / len(test_data) * 100))


def imageShowTest():
    from PIL import Image

    def img_show(img):
        pil_img = Image.fromarray(np.uint8(img))
        pil_img.show()

    (x_train, t_train), _ = load_mnist(flatten=False, normalize=False)

    img = x_train[32]
    label = t_train[32]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)


if __name__ == '__main__':
    predictTest()