# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import random as ra


# 학습에 필요한 상수 정의
_N_FEATURE = 2 # 입력 샘플의 특징 갯수
_N_CLASS = 1 # 분류해야하는 CLASS 갯수 ( 분류결과가 1이 아니면 무조건 0 이기때문에 CLASS 갯수는 1개로 함)
_N_HIDDENLAYER_UNIT = 2 # Hidden Layer 의 Perceptron Unit 갯수


# Activation 함수로 사용하는 Logistic Function    
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

    
# 학습 샘플들의 특징들이 담긴 리스트를 받아 맨 앞에 bias을 위한 1을 추가해서 행렬로 저장해 리턴해주는 함수
def getInputMatrix(dataList):
    featureMatrix = []
    
    for i in range(len(dataList)):
        data = np.array(dataList[i]).astype('float32')
        data = np.insert(dataList[i], 0, 1.0)
        featureMatrix = np.append(featureMatrix, data)
        
    featureMatrix = featureMatrix.reshape(len(dataList), -1)
    
    return featureMatrix
    
# MLP의 결과와 실제 정답을 받아서 분류결과의 정답여부를 리턴해주는 함수
def getTestResult(oValue, trLabelList):
    correctCnt = 0
    wrongCnt = 0
    
    for i in range(len(oValue[0])):
        if oValue[0][i] > 0.5:
            if trLabelList[i] == 1:
                correctCnt += 1
            else:
                wrongCnt += 1
        else:
            if trLabelList[i] == 1:
                wrongCnt += 1
            else:
                correctCnt += 1
            
    return correctCnt, wrongCnt


# Perceptron 클래스
class Perceptron:
    
    def __init__(self, numParam):
        # 1개 + 특징갯수 만큼의 weightList 의 원소 랜덤생성
        self.weightList = []
        for weight in range(1+numParam):
            self.weightList.append(ra.uniform(-1.0, 1.0))
        self.weightList = np.array(self.weightList)

            
# MLP(Multi-Layer Perceptron) 클래스
class MultiLayerPerceptron:

    def __init__(self, trDataList, trLabelList):
       
        ### 반복횟수 변수 생성, 학습률과 랜덤시드 선언
        trainCnt = 0
        learningRate = 1.0
        ra.seed(12345)
        
        # Hidden Layer 생성
        self.hiddenLayer = []
        for i in range(_N_HIDDENLAYER_UNIT):
            self.hiddenLayer.append(Perceptron(_N_FEATURE))
            
        # Output Layer 생성
        self.outputLayer = []
        for i in range(_N_CLASS):
            self.outputLayer.append(Perceptron(_N_HIDDENLAYER_UNIT))

        # 학습데이터들의 featureMatrix 행렬과 정답레이블을 가지고 있는 dValue 배열 생성
        featureMatrix = getInputMatrix(trDataList)
        dValue = np.array(trLabelList).astype('float32').reshape(_N_CLASS, len(trDataList))
        
        # 학습 과정 로그파일 생성
        fdTrainLog = open('train_log.txt', 'w')
        

        # 학습종료 조건이 만족할 때 까지 매 epoch 을 반복
        while True:
            #----- Foward Propagation 시작 -----#

            ## Hidden Layer에 대해 진행해서 각 샘플에 대한 h값을 생성
            # 각 퍼셉트론 의 weight 가져와서 matrix 생성
            hiddenLayerWeightList = [] 
            for i in range(_N_HIDDENLAYER_UNIT):
                hiddenLayerWeightList.append(self.hiddenLayer[i].weightList)
            hiddenLayerWeightMatrix = np.array(hiddenLayerWeightList).reshape(_N_HIDDENLAYER_UNIT, _N_FEATURE+1)

            # dot product 후 logistic function 으로 h값 행렬을 생성 
            hValue = logistic( np.dot(hiddenLayerWeightMatrix, featureMatrix.T).reshape(_N_HIDDENLAYER_UNIT, len(trDataList)) )

            
            ## Output Layer에 대해 진행해서 각 샘플에 대한 o값을 생성
            # 각 퍼셉트론 의 weight 가져와서 matrix 생성
            outputLayerWeightList = [] 
            for i in range(_N_CLASS):
                outputLayerWeightList.append(self.outputLayer[i].weightList)
            outputLayerWeightMatrix = np.array(outputLayerWeightList).reshape(_N_CLASS, _N_HIDDENLAYER_UNIT+1)

            # Hidden Layer 를 거쳐 나온 값들에 1을 추가해서 다시 행렬형태로 정리
            secondFeatureMatrix = np.array([])
            for i in range(len(trDataList)):
                secondFeatureMatrix = np.insert(secondFeatureMatrix, i, np.insert(hValue.T[i,:], 0, 1.0))
            secondFeatureMatrix = secondFeatureMatrix.reshape(len(trDataList), 1+_N_HIDDENLAYER_UNIT)
                        
            # dot product 후 logistic function 으로 o값 행렬을 생성
            oValue = logistic( np.dot(outputLayerWeightMatrix, secondFeatureMatrix.T).reshape(_N_CLASS, len(trDataList)) )
            
            #----- Foward Propagation 종료 -----#
            
            
            # 오류율 검사 및 로그파일 저장
            correctCnt, wrongCnt = getTestResult(oValue, trLabelList)
            errorRate = 100.0*wrongCnt/(correctCnt+wrongCnt)
            print '%3d-th training errorRate : %.2f%% (%d / %d)' % (trainCnt, errorRate, wrongCnt, correctCnt+wrongCnt)
            fdTrainLog.write('%3d-th training errorRate : %.2f%% (%d / %d)\n' % (trainCnt, errorRate, wrongCnt, correctCnt+wrongCnt))
            
            # 오류율이 0%에 도달했으면 반복문 탈출
            if errorRate == 0:
                fdTrainLog.close()
                break;
            
            
            
            #----- Backward Propagation 시작 -----#
            
            # Output Layer의 변화율 계산
            outputLayerDeltaW = np.zeros(_N_CLASS * (1+_N_HIDDENLAYER_UNIT)).reshape(_N_CLASS, 1+_N_HIDDENLAYER_UNIT)          
            outputLayerUpdateValue = (dValue-oValue) * oValue * (1.0-oValue) # _N_CLASS x 데이터수 행렬
            outputLayerDeltaW += (-1.0) * learningRate * np.dot(outputLayerUpdateValue, secondFeatureMatrix)

            
            # Hidden Layer의 변화율 계산
            hiddenLayerDeltaW = np.zeros(_N_HIDDENLAYER_UNIT * (1+_N_FEATURE)).reshape(_N_HIDDENLAYER_UNIT, 1+_N_FEATURE)

            # 각 샘플들에 대해서 반복
            for bs in range(len(trDataList)): 
                colSum = np.sum(outputLayerUpdateValue[0, bs] * outputLayerWeightMatrix, axis=0).reshape(_N_CLASS, 1+_N_HIDDENLAYER_UNIT)
                hiddenLayerUpdateValue = []
                
                # Hidden Layer의 퍼셉트론들에 대해 각각 계산해서 배열에 저장하고, 이를 모아 행렬로 만들어 한번에 deltaW에 더해줌
                for i in range(_N_HIDDENLAYER_UNIT): 
                    calc = -1.0 * learningRate * colSum[0, i] * hValue[i, bs] * (1.0-hValue[i, bs]) * featureMatrix[bs, :]  ### 다 스칼라값이고 마지막 featureMatrix[i, :] 만 벡터(행렬x)고 길이는 3 (1+_N_FEATURE)
                    hiddenLayerUpdateValue.append(calc)
                hiddenLayerDeltaW += np.array(hiddenLayerUpdateValue).reshape(_N_HIDDENLAYER_UNIT, 1+_N_FEATURE)

                
            # Output Layer의 파라메터 업데이트
            for i in range(_N_CLASS):
                self.outputLayer[i].weightList -= outputLayerDeltaW[i,:]
                
            # Hidden Layer의 파라메터 업데이트
            for i in range(_N_HIDDENLAYER_UNIT):
                self.hiddenLayer[i].weightList -= hiddenLayerDeltaW[i,:]
                
            #----- Backward Propagation 종료 -----#
                
            trainCnt += 1
            
            

    
    
    
# 프로그램 메인함수    
if __name__ == '__main__':
    
    # 학습 데이터와 정답 레이블 생성
    xorDataList = [ [0,0], [0,1], [1,0], [1,1] ]
    xorLabelList = [ 0, 1, 1, 0 ]
    
    # Multi-Layer Perceptron 을 만들어서 학습
    mlp = MultiLayerPerceptron(xorDataList, xorLabelList)
    
    # 테스트 로그파일 생성
    fdTestLog = open('test_output.txt', 'w')
    
    # 테스트 데이터들에 1을 추가해서 다시 행렬형태로 정리
    featureMatrix = getInputMatrix(xorDataList)
    
    # Hidden Layer의 각 퍼셉트론 의 weight 가져와서 matrix 생성
    hiddenLayerWeightList = [] 
    for i in range(_N_HIDDENLAYER_UNIT):
        hiddenLayerWeightList.append(mlp.hiddenLayer[i].weightList)
    hiddenLayerWeightMatrix = np.array(hiddenLayerWeightList).reshape(_N_HIDDENLAYER_UNIT, _N_FEATURE+1)

    # dot product 후 logistic function 으로 h값 행렬을 생성 
    hValue = logistic( np.dot(hiddenLayerWeightMatrix, featureMatrix.T).reshape(_N_HIDDENLAYER_UNIT, len(xorDataList)) )

    # Output Laye의 각 퍼셉트론 의 weight 가져와서 matrix 생성
    outputLayerWeightList = [] 
    for i in range(_N_CLASS):
        outputLayerWeightList.append(mlp.outputLayer[i].weightList)
    outputLayerWeightMatrix = np.array(outputLayerWeightList).reshape(_N_CLASS, _N_HIDDENLAYER_UNIT+1)

    # Hidden Layer 를 거쳐 나온 값들에 1을 추가해서 다시 행렬형태로 정리
    secondFeatureMatrix = np.array([])
    for i in range(len(xorDataList)):
        secondFeatureMatrix = np.insert(secondFeatureMatrix, i, np.insert(hValue.T[i,:], 0, 1.0))
    secondFeatureMatrix = secondFeatureMatrix.reshape(len(xorDataList), 1+_N_HIDDENLAYER_UNIT)
    
    # dot product 후 logistic function 으로 o값 행렬을 생성
    oValue = logistic( np.dot(outputLayerWeightMatrix, secondFeatureMatrix.T).reshape(_N_CLASS, len(xorDataList)) )
    
    # 테스트 결과 출력 및 로그저장
    for i in range(len(xorDataList)):
        if oValue[0][i] > 0.5:
            label = 1
        else:
            label = 0
        
        if label == xorLabelList[i]:
            result = 'correct'
        else:
            result = 'wrong'
            
        print 'Test Data: ' + str(xorDataList[i]) + ' -> Classification Result: ' + str(label) + ' (' + result + ')'
        fdTestLog.write('Test Data: ' + str(xorDataList[i]) + ' -> Classification Result: ' + str(label) + ' (' + result + ')\n')
        
    fdTestLog.close()