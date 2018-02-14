# coding: utf-8
import numpy as np


def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용
    
    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]



# 데이터를 뒤섞는다.
# x - 훈련데이터
# t - 훈련데이터 레이블
def shuffle_dataset(x, t):
    # 훈련 데이터의 수까지의 인덱스 배열을 생성 후 뒤섞는다.
    permutation = np.random.permutation(x.shape[0])

    # 입력데이터가 2차원인 경우와 4차원 텐서(합성곱)인 경우 구분
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t




# CNN의 4차원 입력 이미지 데이터들(N, C, H, W)을 하나의 필터의 크기(C, FH, FW) 만큼 잘라서 1차원 벡터의 배열(2차원 행렬)로 풀어서 변환
# 즉, 2차원 배열로 변환된 각 행을 구성하는 열은 입력 데이터를 3차원 필터(C, H, W) 크기 만큼의 값들을 1차원 벡터로 변환한 것이다.
# 행: 이미지 데이터들에 적용되는 필터 연산의 회수 = 전체 Conv 계층의 출력의 수(N * 출력 높이 * 출력 폭)
# 열: 필터의 크기에 해당하는 이미지 데이터 벡터(C * 필터의 높이 * 필터의 폭)
#   input_data : 4차원 배열 형태의 입력 데이터(데이터 수, 채널 수, 높이, 폭)
#   FH : 필터의 높이
#   FW : 필터의 폭
#   stride : 스트라이드
#   pad : 패딩
def im2col(input_data, FH, FW, stride=1, pad=0):
    # 입력의 형상(데이트 수, 채널 수, 높이, 폭)
    N, C, H, W = input_data.shape

    # 출력의 높이, 폭 계산
    OH = int((H + 2 * pad - FH) / stride + 1)
    OW = int((W + 2 * pad - FW) / stride + 1)


    # 입력 이미지들에 패딩 적용(0으로 패딩) - 이미지의 높이(H)와 폭(W) 차원에만 적용
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    # 변환될 2차원 배열을 6차원(데이터 수, 채널, 필터 높이, 필터 폭, 출력 높이, 출력 폭)으로 초기화
    col = np.zeros((N, C, FH, FW, OH, OW))


    # 필터의 크기만큼 입력 데이터 추출(필터의 동일한 인덱스에 위치할 이미지 데이터 추출)
    for filter_y in range(FH):
        y_max = filter_y + stride * OH
        for filter_x in range(FW):
            x_max = filter_x + stride * OW
            col[:, :, filter_y, filter_x, :, :] = img[:, :, filter_y:y_max:stride, filter_x:x_max:stride]


    # (N * OH * OW, C * FH * HW)
    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(N * OH * OW, -1)
    return col





# im2col로 2차원 배열로 변환된 데이터들(N * OH * OW, C * FH * FW)을 원래의 4차원 데이터(N, C, H, W)로 변환
#   col : 2차원 배열(입력 데이터)
#   input_shape : 원래 이미지 데이터의 형상（N, C, H, W)
#   FH : 필터의 높이
#   FW : 필터의 폭
#   stride : 스트라이드
#   pad : 패딩
def col2im(col, input_shape, FH, FW, stride=1, pad=0):
    # 입력의 형상(데이트 수, 채널 수, 높이, 폭)
    N, C, H, W = input_shape

    # 출력의 높이, 폭 계산
    OH = int((H + 2 * pad - FH) / stride + 1)
    OW = int((W + 2 * pad - FW) / stride + 1)


    # im2col로 변환된 2차원 배열을 6차원(N, OH, OW, C, FH, FW)으로 reshape 후,
    # (N, C, FH, FW, OH, OW) 형태로 변환
    col = col.reshape(N, OH, OW, C, FH, FW).transpose((0, 3, 4, 5, 1, 2))

    # 원래대로 복구할 이미지 배열(4차원) 초기화
    # (데이터 수, 채널, 패딩을 포함할 이미지의 높이, 패딩을 포함한 이미지의 폭)
    img = np.zeros((N, C, H + 2*pad, W + 2*pad))


    for filter_y in range(FH):
        y_max = filter_y + stride * OH
        for filter_x in range(FW):
            x_max = filter_x + stride * OW
            img[:, :, filter_y:y_max:stride, filter_x:x_max:stride] = col[:, :, filter_y, filter_x, :, :]

    # 패딩 제거
    return img[:, :, pad:H+pad, pad:W+pad]





if __name__ == '__main__':

    # (데이터 수, 채널 수, 높이, 너비)
    x1 = np.random.rand(10, 3, 7, 7)
    col1 = im2col(x1, 5, 5, stride=3, pad=0)
    x2 = col2im(col1, (10, 3, 7, 7), 5, 5, stride=3, pad=0)
    print(x1 == x2)

