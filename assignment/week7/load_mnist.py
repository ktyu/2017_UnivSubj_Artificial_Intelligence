# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import matplotlib.pyplot as plt



# MNIST 데이터 경로
_SRC_PATH = u'mnist\\raw_binary'
_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images.idx3-ubyte'
_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels.idx1-ubyte'
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL

# 출력 이미지 경로
_DST_PATH = u'img_gray'



def drawImage(dataArr, fn):
    fig, ax = plt.subplots()
    ax.imshow(dataArr, cmap='gray')
    #plt.show()
    plt.savefig(fn)
    
    
    
def loadData(fn):
    print 'loadData', fn
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    nRow = st.unpack('>I', fd.read(4))[0]
    nCol = st.unpack('>I', fd.read(4))[0]
    
    print 'magicNumber', magicNumber
    print 'nData', nData
    print 'nRow', nRow
    print 'nCol', nCol
    
    # data: unsigned byte
    dataList = []
    for i in range(nData):
        dataRawList = fd.read(_N_PIXEL)
        dataNumList = st.unpack('B' * _N_PIXEL, dataRawList)
        dataArr = np.array(dataNumList).reshape(_N_ROW, _N_COL)
        dataList.append(dataArr.astype('int32'))
        
    fd.close()
    
    print 'done.'
    print
    
    return dataList
    


def loadLabel(fn):
    print 'loadLabel', fn
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    
    print 'magicNumber', magicNumber
    print 'nData', nData
    
    # data: unsigned byte
    labelList = []
    for i in range(nData):
        dataLabel = st.unpack('B', fd.read(1))[0]
        labelList.append(dataLabel)
        
    fd.close()
    
    print 'done.'
    print
    
    return labelList



def loadMNIST():
    # 학습 데이터 / 레이블 로드
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)
    
    # 테스트 데이터 / 레이블 로드
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    
    return trDataList, trLabelList, tsDataList, tsLabelList

    
    
if __name__ == '__main__':
    trDataList, trLabelList, tsDataList, tsLabelList = loadMNIST()
    
    print 'len(trDataList)', len(trDataList)
    print 'len(trLabelList)', len(trLabelList)
    print 'len(tsDataList)', len(tsDataList)
    print 'len(tsLabelList)', len(tsLabelList)
    
    if op.exists(_DST_PATH) == False:
        os.mkdir(_DST_PATH)
        
    # 샘플로 5개씩만 출력해보기
    for i in range(5):
        label = trLabelList[i]
        dstFn = _DST_PATH + u'\\tr_%d_label_%d.png' % (i, label)
        #print '%d-th train data: label=%d' % (i, label)
        drawImage(trDataList[i], dstFn)
        
        dstFn = _DST_PATH + u'\\tr_%d_label_%d.txt' % (i, label)
        np.savetxt(dstFn, trDataList[i], fmt='%4d')
        
    for i in range(5):
        label = tsLabelList[i]
        dstFn = _DST_PATH + u'\\ts_%d_label_%d.png' % (i, label)
        #print '%d-th test data: label=%d' % (i, label)
        drawImage(tsDataList[i], dstFn)
        
        dstFn = _DST_PATH + u'\\ts_%d_label_%d.txt' % (i, label)
        np.savetxt(dstFn, tsDataList[i], fmt='%4d')
        