# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


# MNIST 데이터 파일 경로
_SRC_PATH = u'.\\datasets'

_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images.idx3-ubyte'
_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels.idx1-ubyte'
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'

_NMNIST_AWGN_DATASET_FILE = _SRC_PATH + u'\\mnist-with-awgn.mat'
_NMNIST_MB_DATASET_FILE = _SRC_PATH + u'\\mnist-with-motion-blur.mat'
_NMNIST_RC_DATASET_FILE = _SRC_PATH + u'\\mnist-with-reduced-contrast-and-awgn.mat'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL
_N_CLASS = 10

# 출력 이미지 경로
_DST_PATH = u'.\\images\\module_load_data'

# Noise 생성 비율
_NOISE_FACTOR = 0.7 # factor 값이 높을수록 노이즈가 심하게 생김


def loadNMNIST_AWGN_testData():
    data = sio.loadmat(_NMNIST_AWGN_DATASET_FILE)
    tsDataList = data['test_x']
    tsLabelList = data['test_y']
    
    return tsDataList.reshape(len(tsDataList), _N_ROW, _N_COL, 1), tsLabelList.astype('float32')
    
    
def loadNMNIST_MB_testData():
    data = sio.loadmat(_NMNIST_MB_DATASET_FILE)
    tsDataList = data['test_x']
    tsLabelList = data['test_y']
    
    return tsDataList.reshape(len(tsDataList), _N_ROW, _N_COL, 1), tsLabelList.astype('float32')
    
    
def loadNMNIST_RC_testData():
    data = sio.loadmat(_NMNIST_RC_DATASET_FILE)
    tsDataList = data['test_x']
    tsLabelList = data['test_y']
    
    return tsDataList.reshape(len(tsDataList), _N_ROW, _N_COL, 1), tsLabelList.astype('float32')
    

def drawImage(dataArr, fn):
    fig, ax = plt.subplots()
    ax.imshow(dataArr, cmap='gray')
    #plt.show()
    plt.savefig(fn)
    
    # flush
    plt.cla()
    plt.clf()
    plt.close()
    
   
def loadData(fn):
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    nRow = st.unpack('>I', fd.read(4))[0]
    nCol = st.unpack('>I', fd.read(4))[0]
    
    # data: unsigned byte
    dataList = []
    for i in range(nData):
        dataRawList = fd.read(_N_PIXEL)
        dataNumList = st.unpack('B' * _N_PIXEL, dataRawList)
        dataArr = np.array(dataNumList).reshape(_N_ROW, _N_COL, 1)
        dataList.append(dataArr.astype('int32'))
        
    fd.close()
        
    return np.array(dataList)
    


def loadLabel(fn):
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]

    # data: unsigned byte
    labelList = []
    for i in range(nData):
        dataLabel = st.unpack('B', fd.read(1))[0]
        label = np.zeros(_N_CLASS)
        label[dataLabel] = 1.0
        labelList.append(label)
        
    fd.close()
    
    return np.array(labelList)

def loadMNIST_pure_trainData():
    # 학습 데이터 / 레이블 로드
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)

    return trDataList, trLabelList

    
def loadMNIST_pure_testData():
    # 테스트 데이터 / 레이블 로드
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    
    return tsDataList, tsLabelList
    
def loadMNIST_noise_trainData():
    # 학습 데이터 / 레이블 로드
    pure_trDataList, trLabelList = loadMNIST_pure_trainData()
    
    # 노이즈를 섞음
    noise_trDataList = pure_trDataList + _NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=pure_trDataList.shape) 
    noise_trDataList = np.clip(noise_trDataList, 0., 1.)

    return noise_trDataList, trLabelList


def loadMNIST_noise_testData():
    # 테스트 데이터 / 레이블 로드
    pure_tsDataList, tsLabelList = loadMNIST_pure_testData()
    
    # 노이즈를 섞음
    noise_tsDataList = pure_tsDataList + _NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=pure_tsDataList.shape) 
    noise_tsDataList = np.clip(noise_tsDataList, 0., 1.)
    
    return noise_tsDataList, tsLabelList


# 모듈을 직접 실행시에만 동작하는 메인함수    
if __name__ == '__main__':
    # random seed 설정
    np.random.seed(12345)
    
    # 학습 데이터 로드 (정상 데이터, 노이즈가 섞인 데이터)
    pure_trDataList, trLabelList = loadMNIST_pure_trainData()
    noise_trDataList, trLabelList = loadMNIST_noise_trainData()
    
    # 테스트 데이터 로드 (정상 데이터, 노이즈가 섞인 데이터)
    pure_tsDataList, tsLabelList = loadMNIST_pure_testData()
    noise_tsDataList, tsLabelList = loadMNIST_noise_testData()
    
    print 'MNIST Data load done.\n'

    # 샘플로 5개씩만 출력해보기
    if op.exists(_DST_PATH) == False:
        os.mkdir(_DST_PATH)
    
    if op.exists(_DST_PATH + u'\\data_samples') == False:
        os.mkdir(_DST_PATH + u'\\data_samples')
        
    for i in range(5):
        label = trLabelList[i].argmax()
        dstFn = _DST_PATH + u'\\data_samples\\pure_tr_%d_label_%d.png' % (i, label)
        drawImage(pure_trDataList[i, :, :, 0], dstFn)
        
    for i in range(5):
        label = trLabelList[i].argmax()
        dstFn = _DST_PATH + u'\\data_samples\\noise_tr_%d_label_%d.png' % (i, label)
        drawImage(noise_trDataList[i, :, :, 0], dstFn)

    for i in range(5):
        label = tsLabelList[i].argmax()
        dstFn = _DST_PATH + u'\\data_samples\\pure_ts_%d_label_%d.png' % (i, label)
        drawImage(pure_tsDataList[i, :, :, 0], dstFn)
        
    for i in range(5):
        label = tsLabelList[i].argmax()
        dstFn = _DST_PATH + u'\\data_samples\\noise_ts_%d_label_%d.png' % (i, label)
        drawImage(noise_tsDataList[i, :, :, 0], dstFn)
    
    print "Images are saved in 'images\\module_load_data' folder."