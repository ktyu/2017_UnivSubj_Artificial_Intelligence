# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import random as ra
import pickle as pkl


# MNIST 데이터 경로
_SRC_PATH = u'..\\'
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL
_N_CLASS = 10 # 숫자 0~9까지 총 10개의 output 종

def loadData(fn):
      
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    nRow = st.unpack('>I', fd.read(4))[0]
    nCol = st.unpack('>I', fd.read(4))[0]
    
    # data: unsigned byte
    
    # 처리속도 향상을 위해 1만개의 데이터를 1천개씩 총 10개의 행렬에 담아서 처리
    featureMatrixDivided = []
    for j in range(10):
        featureMatrix = []
        for i in range(nData/10):
            dataRawList = fd.read(_N_PIXEL)
            dataNumList = st.unpack('B' * _N_PIXEL, dataRawList)
            dataArr = np.array(dataNumList)
            dataArr = np.insert(dataNumList, 0, 255.0) # bias를 위한 값 맨 앞에 추가
            featureMatrix = np.append(featureMatrix, dataArr.astype('float32') / 255.0)
        featureMatrix = featureMatrix.reshape(nData/10, -1)
        featureMatrixDivided.append(featureMatrix)
    
        print '...processing data... ( %5d / %5d )'% ((j+1)*1000, nData)
        
    fd.close()
    return featureMatrixDivided


def loadLabel(fn):

    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
      
    # data: unsigned byte
    labelList = []
    for i in range(nData):
        dataLabel = st.unpack('B', fd.read(1))[0]
        labelList.append(dataLabel)
        
    fd.close()
    return labelList


def loadMNIST():
    # 테스트 데이터 / 레이블 로드
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    return tsDataList, tsLabelList
    
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x*0.1))    

def loadpkl(fn):
    fd = open(fn, 'rb')
    obj = pkl.load(fd)
    fd.close()
    return obj
    
    

class Perceptron:

    def __init__(self):
        # 1개 + 특징갯수 만큼의 weightList 의 원소 랜덤생성
        self.weightList = []
        for weight in range(1+_N_ROW*_N_COL):
            self.weightList.append(ra.uniform(-10.0, 10.0))
        self.weightList = np.array(self.weightList)

    
class SingleLayerPerceptron:
    def __init__(self):
        # SLP의 각 Perceptron 생성 (_N_CLASS 개수만큼 == 10개)
        self.perceptronList = []
        for i in range(_N_CLASS):
            self.perceptronList.append(Perceptron())

    
    
    
def getTestResult(featureMatrixDivided, tsLabelList, slp, fdTestLog):
    correctCnt = 0
    wrongCnt = 0
    
    # 각 퍼셉트론 의 weight 가져와서 matrix 생성 ( (10)x (1+784) 크기의 행렬 )
    weightList = [] 
    for i in range(_N_CLASS):
        weightList.append(slp.perceptronList[i].weightList)
    weightMatrix = np.array(weightList).reshape(_N_CLASS,-1)

    # 10개로 나눠진 특징 행렬에 대해 각각 테스트
    iteration = 0
    divideNum = len(tsLabelList) / len(featureMatrixDivided)
    
    for featureMatrix in featureMatrixDivided:
        # dot product 와 logistic function 으로 o값 행렬을 구함
        dotResult = np.dot(weightMatrix, featureMatrix.T).reshape(_N_CLASS, -1)
        oValue = logistic(dotResult)
        
        # oValue 배열에서 가장 큰 값을 가지고 있는 원소의 인덱스가 정답
        for i in range(divideNum):
            if np.argmax(oValue[:,i]) == tsLabelList[i+iteration]:
                correctCnt += 1
                fdTestLog.write('%5d-th sample -> Expected: %d / Answer: %d (Correct)\n' % (i+1+iteration, np.argmax(oValue[:,i]), tsLabelList[i+iteration]))
            else:
                wrongCnt += 1
                fdTestLog.write('%5d-th sample -> Expected: %d / Answer: %d (Wrong)\n' % (i+1+iteration, np.argmax(oValue[:,i]), tsLabelList[i+iteration]))

        iteration += divideNum
                
    return correctCnt*1.0, wrongCnt*1.0
    
    
    

# 메인함수
if __name__ == '__main__':
    
    # MNIST 데이터셋 로드
    featureMatrixDivided, tsLabelList = loadMNIST()
    
    # 저장된 파라메터 로드
    slp = loadpkl('best_param.pkl')
    
    # 출력 로그파일 생성
    fdTestLog = open('test_output.txt', 'w') 
    
    # 테스트 진행
    correctCnt, wrongCnt = getTestResult(featureMatrixDivided, tsLabelList, slp, fdTestLog)
    print '\nTesting result log file is created! (test_output.txt)'
    
    # 최종결과만 콘솔출력
    print 'Final Result -> ErrorRate : %.2f%% ( %5d / %5d )' % ( wrongCnt/(correctCnt+wrongCnt)*100, wrongCnt, correctCnt+wrongCnt )
    fdTestLog.write('Final Result -> ErrorRate : %.2f%% ( %5d / %5d )\n' % ( wrongCnt/(correctCnt+wrongCnt)*100, wrongCnt, correctCnt+wrongCnt ))
    fdTestLog.close()
    
    
