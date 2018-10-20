#-*- coding: utf-8 -*-
import sys
import os.path as op
import random as ra
import copy
import numpy as np


# 실수 데이터가 담긴 txt파일을 인자로 받아서 리스트로 정리해서 리턴해주는 함수
def readFloatTxt(filename):
    fd = open(filename, 'r')
    datalist = fd.readlines()
    fd.close()
        
    for i in range(len(datalist)): 
        datalist[i] = datalist[i].strip('\r\n').split('\t')
        datalist[i] = map(float, datalist[i])
        
    return datalist
    

    
# 파라미터를 기준으로 어종을 분류해주는 함수 (salmon==0 / seabass==1)
# 수식 : A + Bx1+ Cx2 : data->[x1,x2] / param->[A,B,C]
def getClass(data, param):
	res = param[0] + param[1] * data[0] + param[2] * data[1]
	if res < 0:
		return 0
	return 1
    
   
# 오류율을 계산해주는 함수   
def getErrorRate(datalist, param):
    totalCnt = 0.0
    errorCnt = 0.0
    
    for data in datalist:
        if getClass(data, param) == data[-1]:
            totalCnt += 1.0
        else:
            totalCnt += 1.0
            errorCnt += 1.0
    
    return errorCnt / totalCnt
    

    
#####Perceptron을 구현한 클래스
class Perceptron:

    def __init__(self, learningRate):
    
        # 입력파일을 읽어 리스트로 정리
        dataSalmonTrain = readFloatTxt('salmon_train.txt')
        dataSeabassTrain = readFloatTxt('seabass_train.txt')
        
        # Training Data 리스트 마지막에 실제 어종을 추가하고(salmon==0 / seabass==1) 하나의 데이터 셋으로 통합
        for i in range(len(dataSalmonTrain)):
            dataSalmonTrain[i].append(0)
        for i in range(len(dataSeabassTrain)):
            dataSeabassTrain[i].append(1)
        dataTrain = dataSalmonTrain+dataSeabassTrain
        
        # 학습결과 출력 로그파일 생성
        trResFn = 'train_log_%.2f.txt' % (learningRate)
        fdTrainLog = open(trResFn, 'w') 

        # Perceptron의 파라메터 초기값을 랜덤하게 부여
        self.param = []
        for i in range(len(dataTrain[0])): # 입력특징의 갯수(weight) + 1(bias) 인데, 분류결과가 추가되어있으므로 dataTrain 원소 하나의 길이만큼 파라메터 생성
            self.param.append(ra.uniform(-1000.0, 1000.0))
            
        # 초기 Perceptron의 파라메터와 그 에러율을 구하고 출력
        self.bestTrainCnt = 0
        self.bestParam = copy.deepcopy(self.param)
        self.bestErrorRate = getErrorRate(dataTrain, self.bestParam)
        initialMsg = 'Initial Perceptron Parameter\'s Error Rate: %.1f%%' % (self.bestErrorRate*100) + ' -> ' + str(self.bestParam) + '\n'
        fdTrainLog.write(initialMsg + '\n')
        print initialMsg

       
        # 반복문에 필요한 변수들 선언
        stopCond = False
        trainCnt = 0
        
        # while 문으로 반복 학습 시작
        print '-----Training start-----'
        while(not stopCond):
            for data in dataTrain:
                sortingResult = getClass(data[0:-1], self.param)
               
                self.param[0] = self.param[0] + learningRate*(data[-1] - sortingResult)*1 #bias 업데이트 (data[-1] 값은 실제 어종임)
                for i in range(1,len(self.param)):
                    self.param[i] = self.param[i] + learningRate*(data[-1] - sortingResult)*data[i-1] # weight 값들 업데이트
                
                trainCnt += 1
                errorRate = getErrorRate(dataTrain, self.param)      
                
                # 학습결과 로그저장 및 콘솔출력                    
                trainLogSentence = '%5d-th training, ErrorRate: %.1f%%' % (trainCnt, errorRate*100) + ' -> ' + str(self.param)
                fdTrainLog.write(trainLogSentence + '\n')
                print trainLogSentence
                
                # 지금까지 학습 중 최상의 결과(에러율이 가장 낮은 파라메터)를 저장
                if errorRate <= self.bestErrorRate:
                    self.bestTrainCnt = copy.deepcopy(trainCnt)
                    self.bestParam = copy.deepcopy(self.param)
                    self.bestErrorRate = copy.deepcopy(errorRate)
        
                # 정지조건 : 에러율 10% 이하의 파라메터를 찾았거나, 10000개 데이터 이상을 학습한 경우
                if(self.bestErrorRate <= 0.10 or trainCnt >= 10000):
                    stopCond = True
                    break;
         #while 문 종료   


        # 최종결과 로그저장 및 결과 콘솔 출력
        finalMsg = '\nFinal Perceptron Parameter\'s Error Rate (%d-th training): %.1f%% -> '  % (self.bestTrainCnt, self.bestErrorRate*100) + str(self.bestParam) + '\n'
        fdTrainLog.write(finalMsg + '\n')
        print finalMsg, '\nTraining result file: ' + trResFn + '\n\n'
        fdTrainLog.close()        
        
        
       

##### 프로그램 메인함수    
if __name__ == '__main__':
    argmentNum = len(sys.argv)
    
    if argmentNum == 2:  #command line argument 수 확인
        learningRate = float(sys.argv[1])  
        
        # Perceptron 객체를 생성해서 학습진행
        trainingData = Perceptron(learningRate)
        
        # Test 데이터를 읽어들임
        dataSalmonTest = readFloatTxt('salmon_test.txt')
        dataSeabassTest = readFloatTxt('seabass_test.txt')
        
        # 분류결과를 저장할 로그파일 생성
        tsResFn = 'test_output_%.2f.txt' % (learningRate)
        fdTestLog = open(tsResFn, 'w') 
        print '-----Testing start-----'
        
        # salmon 데이터 분류
        for data in dataSalmonTest:
            data.append(getClass(data, trainingData.bestParam))
            if  data[-1] == 0:
                judgeMsg = 'salmon (correct)'
            else:
                judgeMsg = 'seabass (wrong)'
                data[-1] = 0  # 최종 ErrorRate 계산을 위해 잘못 분류된 결과는 정확한 결과로 수정
        
            testResultMsg = 'body: %.1f tail: %.1f ==> %s' % (data[0], data[1], judgeMsg)
            fdTestLog.write(testResultMsg + '\n')
            print testResultMsg
        
        # seabass 데이터 분류
        for data in dataSeabassTest:
            data.append(getClass(data, trainingData.bestParam))
            if data[-1] == 1:
                judgeMsg = 'seabass (correct)'
            else:
                judgeMsg = 'salmon (wrong)'
                data[-1] = 1  # 최종 ErrorRate 계산을 위해 잘못 분류된 결과는 정확한 결과로 수정
        
            testResultMsg = 'body: %.1f tail: %.1f ==> %s' % (data[0], data[1], judgeMsg)
            fdTestLog.write(testResultMsg + '\n')
            print testResultMsg
        
        
        #최종 테스트결과 정확도 로그저장 및 콘솔출력
        finalResultErrorRate = getErrorRate(dataSalmonTest+dataSeabassTest, trainingData.bestParam)
        finalResultMsg = '\nTest data ErrorRate (with Best param): %.1f' % (finalResultErrorRate*100) + '%\n'
        fdTestLog.write(finalResultMsg + '\n')
        print finalResultMsg, '\nTesting result file: ' + tsResFn
        fdTestLog.close()
       
        
    else:
        print 'Usage: %s [LearningRate]' % (op.basename(sys.argv[0]))
          
        
        