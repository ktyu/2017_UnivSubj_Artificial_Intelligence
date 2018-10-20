# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import random as ra
import pickle as pkl


# MNIST 데이터 경로
_SRC_PATH = u'..\\'
_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images.idx3-ubyte'
_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels.idx1-ubyte'
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL


# 학습에 필요한 상수 정의
_N_CLASS = 10 # 숫자 0~9까지 총 10개의 output 종
_BATCH_SIZE = 50 # mini-batch 의 크기


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
        dataArr = np.array(dataNumList)
        dataList.append(dataArr.astype('float32') / 255.0)
        
    fd.close()
    return dataList


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
    # 학습 데이터 / 레이블 로드
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)
    
    return trDataList, trLabelList

# Activation 함수 (logistic function 약간 변형)
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x*0.1))
    
    
def savepkl(fn, obj):
    fd = open(fn, 'wb')
    pkl.dump(obj, fd)
    fd.close()
    
    
# Perceptron 클래스
class Perceptron:

    def __init__(self):
        # 1개 + 특징갯수 만큼의 weightList 의 원소 랜덤생성
        self.weightList = []
        for weight in range(1+_N_ROW*_N_COL):
            self.weightList.append(ra.uniform(-10.0, 10.0))
        self.weightList = np.array(self.weightList)
            
            
            
            
# SLP(Single Layer Perceptron) 클래스
class SingleLayerPerceptron:

    def __init__(self, trDataList, trLabelList):
        # 학습반복 횟수
        trainCnt = 0;
        
        # 초기 학습률
        learningRate = 30.0
        
        # SLP의 각 Perceptron 생성 (_N_CLASS 개수만큼 == 10개)
        self.perceptronList = []
        for i in range(_N_CLASS):
            self.perceptronList.append(Perceptron())

        # validation 을 위해 인덱스 관리
        sampleIndexList = range(len(trDataList))
        ra.shuffle(sampleIndexList)
        validationSetSize = 1000    # validation set 의 사이즈
        trainingSetSize = (len(trDataList) - validationSetSize) / _BATCH_SIZE
        
        # 랜덤하게 섞인 인덱스 리스트로부터 validation set 과 training set 을 분리
        self.validationSampleIndex = sampleIndexList[0:validationSetSize]
        trainingSampleIndex = np.array(sampleIndexList[validationSetSize:]).reshape(-1,_BATCH_SIZE)
        
        # validation set 샘플들의 특징을 행렬로 저장
        featureMatrix = []
        for i in range(validationSetSize):
            featureMatrix = np.append(featureMatrix, np.insert(trDataList[self.validationSampleIndex[i]], 0, 1.0))
        featureMatrix = featureMatrix.reshape(validationSetSize, -1)
        
        
        # 로그파일 생성
        fdTrainLog = open('train_log.txt', 'w') 
        
        # 학습 시작 전 오류율 검사
        correctCnt, wrongCnt = self.getTestResult(featureMatrix)
        print 'Initial -> ErrorRate: %.1f%% - ( %4d / %4d )\n' % (wrongCnt/(correctCnt+wrongCnt)*100 , wrongCnt, correctCnt+wrongCnt)
        fdTrainLog.write('Initial -> ErrorRate: %.1f%% - ( %4d / %4d )\n\n' % (wrongCnt/(correctCnt+wrongCnt)*100 , wrongCnt, correctCnt+wrongCnt))
        
        
        # 학습과정 반복
        while (correctCnt/(correctCnt+wrongCnt) < 0.85) & (trainCnt <= 10000):
            
            # 1 epoch만큼의 학습을 진행해서 델타w 행렬을 리턴받음
            deltaW = self.oneEpochTraining(trDataList, trLabelList, learningRate, trainingSampleIndex[(trainCnt % trainingSetSize), :])
            
            # 델타w 행렬의 값으로 각 Perceptron 의 weight 를 업데이트 해줌
            for i in range(_N_CLASS):
                self.perceptronList[i].weightList += deltaW[i,:]
            
            # 300번 학습할때마다 학습률을 98%로 감소
            if (trainCnt % 300 == 0):
                learningRate *= 0.98
            
            # 50번 학습할때마다 테스트용 샘플에 대해 오류율 검사
            if (trainCnt % 50 == 0):
                correctCnt, wrongCnt = self.getTestResult(featureMatrix)
                print '%5d-th training -> ErrorRate: %.1f%% - ( %4d / %4d )' % (trainCnt, wrongCnt/(correctCnt+wrongCnt)*100 , wrongCnt, correctCnt+wrongCnt)
                fdTrainLog.write('%5d-th training -> ErrorRate: %.1f%% - ( %4d / %4d )\n' % (trainCnt, wrongCnt/(correctCnt+wrongCnt)*100 , wrongCnt, correctCnt+wrongCnt))
        
            # 학습횟수 증가
            trainCnt += 1
         
        fdTrainLog.close()   
            
            
    def getTestResult(self, featureMatrix):
        correctCnt = 0
        wrongCnt = 0
        
        # 각 퍼셉트론 의 weight 가져와서 matrix 생성 ( (10)x (1+784) 크기의 행렬 )
        weightList = [] 
        for i in range(_N_CLASS):
            weightList.append(self.perceptronList[i].weightList)
        weightMatrix = np.array(weightList).reshape(_N_CLASS,-1)


        # dot product 와 logistic function 으로 o값 행렬을 구함
        dotResult = np.dot(weightMatrix, featureMatrix.T).reshape(_N_CLASS, len(self.validationSampleIndex)) # 결과는 10 x validation set크기 의 행렬
        oValue = logistic(dotResult)

        
        # oValue 배열에서 가장 큰 값을 가지고 있는 원소의 인덱스가 정답
        for i in range(len(self.validationSampleIndex)):
            
            if np.argmax(oValue[:,i]) == trLabelList[self.validationSampleIndex[i]]:
                correctCnt += 1
            else:
                wrongCnt += 1
                
        return correctCnt*1.0, wrongCnt*1.0
        
        

        
    # 설정된 Batch Size 만큼의 샘플로 학습을 하고 결과로 나온 델타w값을 행렬로 리턴해주는 함수
    def oneEpochTraining(self, trDataList, trLabelList, learningRate, trainingSampleIndex):
        
        # 델타w 배열 생성해서 초기화
        deltaW = np.zeros(_N_CLASS*(1+_N_COL*_N_ROW)).reshape(_N_CLASS, 1+_N_COL*_N_ROW)
        
        # 각 퍼셉트론 의 weight 가져와서 matrix 생성 ( (10)x (1+784) 크기의 행렬 )
        weightList = [] 
        for i in range(_N_CLASS):
            weightList.append(self.perceptronList[i].weightList)
        weightMatrix = np.array(weightList).reshape(_N_CLASS,-1)

        # 랜덤선택되서 넘겨받은 Batch Size 크기의 1차원배열 trainingSampleIndex 들로 featureMatrix 생성
        featureMatrix = []
        for i in range(_BATCH_SIZE):
            featureMatrix = np.append(featureMatrix, np.insert(trDataList[trainingSampleIndex[i]], 0, 1.0))
        featureMatrix = featureMatrix.reshape(_BATCH_SIZE, -1)
        
        # dot product 와 Activation function 으로 o값 행렬을 구함
        dotResult = np.dot(weightMatrix, featureMatrix.T).reshape(_N_CLASS, _BATCH_SIZE)
        oValue = logistic(dotResult)
        
        # 델타w 행렬을 업데이트 (Batch Size 만큼 반복)
        for bs in range(_BATCH_SIZE):
            
            # one-hot representation 을 이용해서 정답레이블을 표시한 dValue 배열 생성
            dValue = np.zeros(_N_CLASS).astype('float32')
            dValue[trLabelList[trainingSampleIndex[bs]]] = 1.0
            
            # deltaW 행렬 업데이트 (dot product에 앞서 행렬의 row, col 수를 맞추기위해 임시함수 사용)
            temp = (dValue - oValue[:,bs]) * (0.1 * oValue[:,bs]) * (0.1 - 0.1*oValue[:,bs]) # Activation 함수(g(z))를 변경하였으므로 이를 다시 미분하여 수식을 구함
            temp2 = featureMatrix[bs,:]
            
            temp = temp.reshape(-1,1)
            temp2 = temp2.reshape(1,-1)
            
            deltaW += learningRate * np.dot(temp, temp2).reshape(_N_CLASS, 1+_N_ROW*_N_COL)

        return deltaW
    
    
    
    
# 프로그램 메인함수    
if __name__ == '__main__':
    
    # MNIST 데이터셋 로드
    trDataList, trLabelList = loadMNIST()

    # 랜덤시드 설정
    ra.seed(1235678)
    
    # Single Layer Perceptron 클래스를 만들어 학습 진행
    slp = SingleLayerPerceptron(trDataList, trLabelList)
    
    # 학습완료 된 결과를 pkl 파일로 저장
    savepkl( 'best_param.pkl', slp)

