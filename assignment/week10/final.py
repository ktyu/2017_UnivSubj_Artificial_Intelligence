# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import random as ra

# 학습에 필요한 상수 정의
_N_FEATURE = 2 # 입력 샘플의 특징 갯수
_N_1TH_HIDDENLAYER_UNIT = 10 # 1번째 Hidden Layer 의 unit 갯수 (bias를 위한 1은 제외한 unit의 갯수)
_N_2TH_HIDDENLAYER_UNIT = 5 # 2번째 Hidden Layer 의 unit 갯수 (bias를 위한 1은 제외한 unit의 갯수)
_N_3TH_HIDDENLAYER_UNIT = 4 # 3번째 Hidden Layer 의 unit 갯수 (bias를 위한 1은 제외한 unit의 갯수)
_N_CLASS = 1 # 분류해야하는 CLASS 갯수(Output Layer의 unit 갯수) ( XOR의 경우 분류결과 Label이 1이 아니면 무조건 0 이기때문에 1개로 설정)


# Logistic Function    
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

# Relu Fuction
def relu(x):
    return np.maximum(0.0, x)

# Relu Function 미분, 적용 후 astype('float32') 적용해주어야함
def reluPrime(x):
    return (x > 0.0)
    
    

# Perceptron 클래스
class Perceptron:
    
    def __init__(self, numParam):
        # 1개 + 특징갯수 만큼의 weightList 의 원소 랜덤생성
        self.weightList = []
        for weight in range(1+numParam):
            self.weightList.append(ra.uniform(-1.0, 1.0))
        self.weightList = np.array(self.weightList)

        

# DNN(Deep Neural Network) 클래스
class DeepNeuralNetwork:

    def __init__(self, trDataList, trLabelList):
       
        ### 반복횟수 변수 생성, 학습률과 랜덤시드 선언
        trainCnt = 0
        learningRate = 0.1
        ra.seed(12340)
        
        # 1번째 Hidden Layer 생성            
        self.firstHiddenLayer = []
        for i in range(_N_1TH_HIDDENLAYER_UNIT):
            self.firstHiddenLayer.append(Perceptron(_N_FEATURE))
            
        # 2번째 Hidden Layer 생성            
        self.secondHiddenLayer = []
        for i in range(_N_2TH_HIDDENLAYER_UNIT):
            self.secondHiddenLayer.append(Perceptron(_N_1TH_HIDDENLAYER_UNIT))

        # 3번째 Hidden Layer 생성            
        self.thirdHiddenLayer = []
        for i in range(_N_3TH_HIDDENLAYER_UNIT):
            self.thirdHiddenLayer.append(Perceptron(_N_2TH_HIDDENLAYER_UNIT))
            
        # Output Layer 생성
        self.outputLayer = []
        for i in range(_N_CLASS):
            self.outputLayer.append(Perceptron(_N_3TH_HIDDENLAYER_UNIT))

        # 정답레이블을 가지고 있는 dValue 배열 생성
        dValue = np.array(trLabelList).astype('float32').reshape(_N_CLASS, len(trDataList))
        
        # 학습 과정 로그파일 생성
        fdTrainLog = open('train_log.txt', 'w')
        

        # 학습종료 조건이 만족할 때 까지 매 epoch 을 반복
        while True:
            
            #----- Foward Propagation 시작 -----#
            
            # dataList 에 담긴 각 샘플들의 특징 맨 앞에 bias를 위한 1.0을 추가해서 행렬로 저장
            featureMatrix = []
            for i in range(len(trDataList)):
                data = np.array(trDataList[i]).astype('float32')
                data = np.insert(trDataList[i], 0, 1.0)
                featureMatrix = np.append(featureMatrix, data)
            featureMatrix = featureMatrix.reshape(len(trDataList), 1+_N_FEATURE)

            
            # 1번째 Hidden Layer의 각 퍼셉트론 의 weight 가져와서 matrix 생성
            firstHiddenLayerWeightList = [] 
            for i in range(_N_1TH_HIDDENLAYER_UNIT):
                firstHiddenLayerWeightList.append(self.firstHiddenLayer[i].weightList)
            firstHiddenLayerWeightMatrix = np.array(firstHiddenLayerWeightList).reshape(_N_1TH_HIDDENLAYER_UNIT, 1+_N_FEATURE)

            # dot product 후 relu function 으로 h1값 행렬을 생성
            dot1 = np.dot(firstHiddenLayerWeightMatrix, featureMatrix.T).reshape(_N_1TH_HIDDENLAYER_UNIT, len(trDataList))
            h1Value = relu(dot1)
            
            # h1Value 값들에 1을 추가해서 다시 행렬형태로 정리
            h1ValueMatrix = np.array([])
            for i in range(len(trDataList)):
                h1ValueMatrix = np.insert(h1ValueMatrix, i, np.insert(h1Value.T[i,:], 0, 1.0))
            h1ValueMatrix = h1ValueMatrix.reshape(len(trDataList), 1+_N_1TH_HIDDENLAYER_UNIT)
            
            
            
            
            
            # 2번째 Hidden Layer의 각 퍼셉트론 의 weight 가져와서 matrix 생성
            secondHiddenLayerWeightList = [] 
            for i in range(_N_2TH_HIDDENLAYER_UNIT):
                secondHiddenLayerWeightList.append(self.secondHiddenLayer[i].weightList)
            secondHiddenLayerWeightMatrix = np.array(secondHiddenLayerWeightList).reshape(_N_2TH_HIDDENLAYER_UNIT, 1+_N_1TH_HIDDENLAYER_UNIT)

            # dot product 후 relu function 으로 h2값 행렬을 생성
            dot2 = np.dot(secondHiddenLayerWeightMatrix, h1ValueMatrix.T).reshape(_N_2TH_HIDDENLAYER_UNIT, len(trDataList))
            h2Value = relu( dot2 )            

            # h2Value 값들에 1을 추가해서 다시 행렬형태로 정리
            h2ValueMatrix = np.array([])
            for i in range(len(trDataList)):
                h2ValueMatrix = np.insert(h2ValueMatrix, i, np.insert(h2Value.T[i,:], 0, 1.0))
            h2ValueMatrix = h2ValueMatrix.reshape(len(trDataList), 1+_N_2TH_HIDDENLAYER_UNIT)
            
            
            
            
            
            # 3번째 Hidden Layer의 각 퍼셉트론 의 weight 가져와서 matrix 생성
            thirdHiddenLayerWeightList = [] 
            for i in range(_N_3TH_HIDDENLAYER_UNIT):
                thirdHiddenLayerWeightList.append(self.thirdHiddenLayer[i].weightList)
            thirdHiddenLayerWeightMatrix = np.array(thirdHiddenLayerWeightList).reshape(_N_3TH_HIDDENLAYER_UNIT, 1+_N_2TH_HIDDENLAYER_UNIT)

            # dot product 후 relu function 으로 h3값 행렬을 생성
            dot3 = np.dot(thirdHiddenLayerWeightMatrix, h2ValueMatrix.T).reshape(_N_3TH_HIDDENLAYER_UNIT, len(trDataList))
            h3Value = relu( dot3 )            
            
            # h3Value 값들에 1을 추가해서 다시 행렬형태로 정리
            h3ValueMatrix = np.array([])
            for i in range(len(trDataList)):
                h3ValueMatrix = np.insert(h3ValueMatrix, i, np.insert(h3Value.T[i,:], 0, 1.0))
            h3ValueMatrix = h3ValueMatrix.reshape(len(trDataList), 1+_N_3TH_HIDDENLAYER_UNIT)
            
            
            
            
            
            # Output Layer의 각 퍼셉트론 의 weight 가져와서 matrix 생성
            outputLayerWeightList = [] 
            for i in range(_N_CLASS):
                outputLayerWeightList.append(self.outputLayer[i].weightList)
            outputLayerWeightMatrix = np.array(outputLayerWeightList).reshape(_N_CLASS, 1+_N_3TH_HIDDENLAYER_UNIT)

            # dot product 후 logistic function 으로 o값 행렬을 생성
            self.oValue = logistic( np.dot(outputLayerWeightMatrix, h3ValueMatrix.T).reshape(_N_CLASS, len(trDataList)) )
            
            #----- Foward Propagation 종료 -----#
            
            
            # 오류율 검사 및 로그파일 저장
            correctCnt, wrongCnt = self.getXORTestResult(self.oValue, trLabelList)
            errorRate = 100.0*wrongCnt/(correctCnt+wrongCnt)
            print '%3d-th training errorRate : %.2f%% (%d / %d) -> ' % (trainCnt, errorRate, wrongCnt, correctCnt+wrongCnt), 'Foward Propagation result: ' + str(self.oValue[0])
            fdTrainLog.write('%3d-th training errorRate : %.2f%% (%d / %d) --> ' % (trainCnt, errorRate, wrongCnt, correctCnt+wrongCnt))
            fdTrainLog.write('Foward Propagation result: ' + str(self.oValue[0]) + '\n')
                        
            # 오류율이 0%에 도달했으면 반복문 탈출(학습종료)
            if errorRate == 0:
                fdTrainLog.close()
                break;
            
            
            #----- Backward Propagation 시작 -----#
            
            ## Output Layer의 변화율 계산
            outputLayerDeltaW = np.zeros(_N_CLASS * (1+_N_3TH_HIDDENLAYER_UNIT)).reshape(_N_CLASS, 1+_N_3TH_HIDDENLAYER_UNIT)          
            outputLayerDeltaSum = (dValue-self.oValue) * self.oValue * (1.0-self.oValue) # _N_CLASS x 데이터수 행렬, logistic function의 미분 적용
            outputLayerDeltaW += (-1.0) * learningRate * np.dot(outputLayerDeltaSum, h3ValueMatrix)


            
            ## 3번째 Hidden Layer의 변화율 계산
            thirdHiddenLayerDeltaW = np.zeros(_N_3TH_HIDDENLAYER_UNIT * (1+_N_2TH_HIDDENLAYER_UNIT)).reshape(_N_3TH_HIDDENLAYER_UNIT, 1+_N_2TH_HIDDENLAYER_UNIT)
            thirdHiddenLayerDeltaSum = []
            
            # 각 샘플(dataidx)들에 대해서 반복
            for dataidx in range(len(trDataList)):

                # data번째 샘플의 오차(output layer의 모든 퍼셉트론의 각 파라메터의 오차 합) 
                thisSampleError = np.sum(outputLayerDeltaSum[:, dataidx].reshape(_N_CLASS, 1) * outputLayerWeightMatrix[:, 1:], axis=0).reshape(_N_3TH_HIDDENLAYER_UNIT, 1)   # _N_3TH_HIDDENLAYER_UNIT x 1 (세로벡터꼴)
                                                    # _N_CLASS x 데이터수에서 한col씩 sliceing               # _N_CLASS x _N_3TH_HIDDENLAYER_UNIT
                thirdHiddenLayerDeltaSum.append(thisSampleError) # 다음 layer의 back propagation 을 위해 저장해놓음
                                         
                thisSampleUpdateValue = (-1.0) * learningRate * thisSampleError * np.dot( reluPrime( dot3[:, dataidx] ).astype('float32').reshape( _N_3TH_HIDDENLAYER_UNIT, 1), h2ValueMatrix[dataidx, :].reshape(1, 1+_N_2TH_HIDDENLAYER_UNIT) )
                                                                                                          # _N_3TH_HIDDENLAYER_UNIT x 1 (세로벡터꼴)                                          # 1 x 1+_N_2TH_HIDDENLAYER_UNIT (가로벡터꼴)
                thirdHiddenLayerDeltaW += thisSampleUpdateValue # data번째 샘플로 부터 얻은 퍼셉트론들의 변화율 저장

                
                
                
            ## 2번째 Hidden Layer의 변화율 계산
            secondHiddenLayerDeltaW = np.zeros(_N_2TH_HIDDENLAYER_UNIT * (1+_N_1TH_HIDDENLAYER_UNIT)).reshape(_N_2TH_HIDDENLAYER_UNIT, 1+_N_1TH_HIDDENLAYER_UNIT)
            secondHiddenLayerDeltaSum = []
            
            # 각 샘플(dataidx)들에 대해서 반복
            for dataidx in range(len(trDataList)):

               # data번째 샘플의 오차(3번째 hidden layer의 모든 퍼셉트론의 각 파라메터의 오차 합)
                thisSampleError = np.sum(thirdHiddenLayerDeltaSum[dataidx]                  *             reluPrime( dot3[:,dataidx].reshape(_N_3TH_HIDDENLAYER_UNIT, 1) )      *      thirdHiddenLayerWeightMatrix[:, 1:], axis=0).reshape(_N_2TH_HIDDENLAYER_UNIT, 1) # _N_2TH_HIDDENLAYER_UNIT x 1 (세로벡터꼴)
                                                  # _N_3TH_HIDDENLAYER_UNITx1 (세로벡터꼴)                             # _N_3TH_HIDDENLAYER_UNIT x 1(세로벡터꼴)                       # _N_3TH_HIDDENLAYER_UNIT x _N_2TH_HIDDENLAYER_UNIT
                secondHiddenLayerDeltaSum.append(thisSampleError) # 다음 layer의 back propagation 을 위해 저장해놓음
                                         
                thisSampleUpdateValue = (-1.0) * learningRate * thisSampleError * np.dot( reluPrime( dot2[:, dataidx] ).astype('float32').reshape( _N_2TH_HIDDENLAYER_UNIT, 1), h1ValueMatrix[dataidx, :].reshape(1, 1+_N_1TH_HIDDENLAYER_UNIT) )
                                                                                                              # _N_2TH_HIDDENLAYER_UNIT x 1 (세로벡터꼴)                                                  # 1 x 1+_N_1TH_HIDDENLAYER_UNIT (가로벡터꼴)
                secondHiddenLayerDeltaW += thisSampleUpdateValue # data번째 샘플로 부터 얻은 퍼셉트론들의 변화율 저장

            
            
            
            ## 1번째 Hidden Layer의 변화율 계산
            firstHiddenLayerDeltaW = np.zeros(_N_1TH_HIDDENLAYER_UNIT * (1+_N_FEATURE)).reshape(_N_1TH_HIDDENLAYER_UNIT, 1+_N_FEATURE)
            
            # 각 샘플(dataidx)들에 대해서 반복
            for dataidx in range(len(trDataList)):

               # data번째 샘플의 오차(2번째 hidden layer의 모든 퍼셉트론의 각 파라메터의 오차 합)
                thisSampleError = np.sum(secondHiddenLayerDeltaSum[dataidx]                  *             reluPrime( dot2[:, dataidx].reshape(_N_2TH_HIDDENLAYER_UNIT, 1) )      *      secondHiddenLayerWeightMatrix[:, 1:], axis=0).reshape(_N_1TH_HIDDENLAYER_UNIT, 1) # _N_1TH_HIDDENLAYER_UNIT x 1 (세로벡터꼴)
                                                  # _N_2TH_HIDDENLAYER_UNITx1 (세로벡터꼴)                             # _N_2TH_HIDDENLAYER_UNIT x 1(세로벡터꼴)                       # _N_2TH_HIDDENLAYER_UNIT x _N_1TH_HIDDENLAYER_UNIT
                                         
                thisSampleUpdateValue = (-1.0) * learningRate * thisSampleError * np.dot( reluPrime( dot1[:, dataidx] ).astype('float32').reshape( _N_1TH_HIDDENLAYER_UNIT, 1), featureMatrix[dataidx, :].reshape(1, 1+_N_FEATURE) )
                                                                                                              # _N_1TH_HIDDENLAYER_UNIT x 1 (세로벡터꼴)                                                  # 1 x 1+_N_FEATURE (가로벡터꼴)
                firstHiddenLayerDeltaW += thisSampleUpdateValue # data번째 샘플로 부터 얻은 퍼셉트론들의 변화율 저장

            
            
            
            
            # Output Layer의 파라메터 업데이트
            for i in range(_N_CLASS):
                self.outputLayer[i].weightList -= outputLayerDeltaW[i,:]
                
            # 3번째 Hidden Layer의 파라메터 업데이트
            for i in range(_N_3TH_HIDDENLAYER_UNIT):
                self.thirdHiddenLayer[i].weightList -= thirdHiddenLayerDeltaW[i,:]
                
            # 2번째 Hidden Layer의 파라메터 업데이트
            for i in range(_N_2TH_HIDDENLAYER_UNIT):
                self.secondHiddenLayer[i].weightList -= secondHiddenLayerDeltaW[i,:]
            
            # 1번째 Hidden Layer의 파라메터 업데이트
            for i in range(_N_1TH_HIDDENLAYER_UNIT):
                self.firstHiddenLayer[i].weightList -= firstHiddenLayerDeltaW[i,:]

                
            #----- Backward Propagation 종료 -----#
                
            trainCnt += 1
            
            
            
            
    # 실제 XOR 연산의 결과를 받아서 학습모델의 결과값과 비교하는 함수
    def getXORTestResult(sefl, oValue, trLabelList):
        correctCnt = 0
        wrongCnt = 0
        
        # o값이 0.5 이상이면 분류결과가 1이고, 0.5 미만이면 분류결과가 1이 아니기 때문에 0임
        for i in range(len(oValue[0,:])):
            if oValue[0,i] >= 0.5:
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
        
        
    
# 프로그램 메인함수    
if __name__ == '__main__':
    
    # 학습 데이터와 정답 레이블 생성
    xorDataList = [ [0,0], [0,1], [1,0], [1,1] ]
    xorLabelList = [ 0, 1, 1, 0 ]
    
    # Multi-Layer Perceptron 을 만들어서 학습하고, 정지조건을 만족한 모델을 가지고있음
    dnn = DeepNeuralNetwork(xorDataList, xorLabelList)
    
    # 테스트 로그파일 생성
    fdTestLog = open('test_output.txt', 'w')
    
    # 학습모델에 저장된 최종 파라메터로 진행한 학습결과와 실제 XOR 계산결과와 비교
    print '\n----------------------Test Start----------------------'
    for i in range(len(xorDataList)):
        if dnn.oValue[0,i] >= 0.5:
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
