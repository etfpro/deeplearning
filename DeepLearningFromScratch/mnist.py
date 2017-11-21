# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np
import functions as f


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


# mnist 이미지 파일을 numpy 배열(60,0000 X 784)로 읽는다.
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


# mnist 파일들을 읽어서 numpy 배열로 변환
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset


# mnist 데이터를 다운받아서 numpy 배열로 변환하여 pickle 파일로 저장
def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


# 레이블 데이터를 10개의 출력으로 one-hot encoding (입력데이터 수 X 10)
def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, data in enumerate(T):
        data[X[idx]] = 1
        
    return T
    

# pickle 파일로 저장된 MNIST 데이터셋 읽어오기
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기
    
    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label : 
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다. 
    
    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    # pickle 파일로 저정된 mnist 데이터를 읽어들인다.
    if not os.path.exists(pickle_path):
        init_mnist()

    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)


    # 입력값을 0.0 ~ 1.0 사이의 실수값으로 정규화
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0


    # on-hot encoding
    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])    


    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

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
    z1 = f.sigmoid(a1)

    # 2 레이어
    a2 = np.dot(z1, W2) + b2
    z2 = f.sigmoid(a2)

    # 3 레이어(출력)
    a3 = np.dot(z2, W3) + b3
    y = f.softmax(a3)

    return y



################################################################################

def predictTest():
    test_data, test_label = getTestData()
    network = initNetwork()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(test_data), batch_size):
        x_batch = test_data[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == test_label[i:i+batch_size])

    print("Accuracy:", accuracy_cnt / len(test_data))


def test():
    test_data, _ = getTestData()
    network = initNetwork()
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]

    print(test_data.shape)
    print(test_data[0].shape)
    print(W1.shape)
    print(W2.shape)
    print(W3.shape)



if __name__ == '__main__':
    predictTest()