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



def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1



# CNN의 입력 이미지 데이터들(4차원 (N, C, H, W))을 2차원 배열로 전개
# 이미지 데이터들을 하나의 필터의 크기(C, H, W) 만큼 잘라서 1차원 벡터로 변환
# 즉, 2차원 배열로 전개된 각 행을 구성하는 열은 입력 데이터를 3차원 필터(C, H, W) 크기 만큼의 값들을 1차원 벡터로 변환한 것이다.
# 행: 이미지 데이터들에 적용되는 필터 연산의 회수 = 전체 Conv 계층의 출력의 수(N * 출력 높이 * 출력 폭)
# 열: 필터의 크기에 해당하는 이미지 데이터 벡터(C * 필터의 높이 * 필터의 폭)
#   input_data : 4차원 배열 형태의 입력 데이터(데이터 수, 채널 수, 높이, 폭)
#   filter_h : 필터의 높이
#   filter_w : 필터의 폭
#   stride : 스트라이드
#   pad : 패딩
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape

    # Conv 계층의 출력의 높이, 폭 계산
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # 입력 이미지들에 패딩 적용(0으로 패딩)
    # 이미지의 높이(H)와 폭(W) 차원에만 적용
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    # 전개될 2차원 배열을 6차원(데이터 수, 채널, 필터 높이, 필터 폭, 출력의 높이, 출력의 폭)으로 초기화
    # 출력
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 필터의 크기만큼
    for filter_y in range(filter_h):
        y_max = filter_y + stride * out_h
        for filter_x in range(filter_w):
            x_max = filter_x + stride * out_w
            col[:, :, filter_y, filter_x, :, :] = img[:, :, filter_y:y_max:stride, filter_x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col



# im2col로 2차원 배열로 전개된 데이터들을 원래의 4차원(N, C, H, W)으로 변환
#   col : 2차원 배열(입력 데이터)
#   input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
#   filter_h : 필터의 높이
#   filter_w : 필터의 폭
#   stride : 스트라이드
#   pad : 패딩
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
