#-*- coding: utf-8 -*-
import sys
import os.path as op
import random as ra
import copy
import numpy as np


# 실수 데이터가 담긴 txt파일을 인자로 받아서 리스트로 정리해서 리턴해주는 함수
def readFloatTxt(filename):
    fd = open(filename, 'r')
    data = fd.readlines()
    fd.close()
        
    for i in range(len(data)): 
        data[i] = data[i].strip('\r\n').split('\t')
        data[i] = map(float, data[i])
        
    return data
    
    
# 파라미터를 기준으로 어종을 분류해주는 함수
def getClass(data, param):
	res = param[0] * data[0] + param[1] * data[1] + param[2]

	if res < 0:
		return 'salmon'
	return 'seabass'
    
   
# 오류율을 계산해주는 함수   
def getErrorRate(salmondata, seabassdata, param):
    totalCnt = 0.0
    errorCnt = 0.0
    
    for data in salmondata:
        if getClass(data, param) == 'salmon':
            totalCnt += 1.0
        else:
            totalCnt += 1.0
            errorCnt += 1.0
    
    for data in seabassdata:
        if getClass(data, param) == 'seabass':
            totalCnt += 1.0
        else:
            totalCnt += 1.0
            errorCnt += 1.0
    
    return errorCnt / totalCnt
    

    
#####Genetic Algorithm 을 구현한 클래스
class GA:

    def __init__(self, popSize, eliteNum, mutProb):
    
        # 입력파일을 읽어 리스트로 정리
        dataSalmonTrain = readFloatTxt('salmon_train.txt')
        dataSeabassTrain = readFloatTxt('seabass_train.txt')
        
        # 학습결과 출력 로그파일 생성
        trResFn = 'train_log_%d_%d_%.2f.txt' % (popSize, eliteNum, mutProb)
        fdTrainLog = open(trResFn, 'w') 

        # 0세대 개체들을 랜덤생성
        self.genePool = []
        for i in range(popSize):
            a = ra.uniform(-100.0, 100.0)
            b = ra.uniform(-100.0, 100.0)
            c = ra.uniform(-1000.0, 1000.0)
            self.genePool.append([a,b,c])
            
        # 0세대의 costList, fitList, bestElite 를 구함
        self.costList = self.getCostList(self.genePool, dataSalmonTrain, dataSeabassTrain)
        self.fitList = self.getFitList(self.costList)
        self.bestElite = self.getBestElite(self.genePool, self.fitList)
        self.bestEliteErrorRate = getErrorRate(dataSalmonTrain, dataSeabassTrain, self.bestElite)*100.0
        
        # 0세대 정보 출력
        initialMsg = 'Initial GenePool BestElite\'s Error Rate: ' + str(self.bestEliteErrorRate) + '%\n'
        fdTrainLog.write(initialMsg + '\n')
        print initialMsg
      
        
        # 돌연변이 발생시 변화 범위 설정
        self.mutationRangeList = [[-50.0, 50.0], [-50.0, 50.0], [-500.0, 500.0]]
        
       
        # 반복문에 필요한 변수들 선언
        trainCnt = 0
        stopCond = False
        self.finalTrainCnt = 0
        self.finalElite = []
        self.finalEliteRate = 100.0
        
        # while 문으로 반복 학습 시작
        print '-----Training start-----'
        while(not stopCond):
            trainCnt += 1
            
            newGenePool = []
            
            # 부모를 selection 해서 자식들을 생성함
            for i in range(popSize-eliteNum):
                papaIndex = self.selection(self.fitList)
                mamaIndex = self.selection(self.fitList)
                offspring = self.makeOffspring(self.genePool, papaIndex, mamaIndex)
                newGenePool.append(offspring)
            
            
            # 생성된 자식들에 돌연변이 조작
            for i in range(len(newGenePool)):
                newGenePool[i] = self.mutation(newGenePool[i], mutProb)
                
                
            # 아직 추가하지 않은 eliteNum 갯수 만큼 elite 들을 추가
            sortedIndexList = np.argsort(self.fitList)
            for i in range(eliteNum):
                newGenePool.append(self.genePool[sortedIndexList[::-1][i]])               
                
                
            # 완성된 다음 세대 유전자 풀로 정보 업데이트
            self.genePool = copy.deepcopy(newGenePool)
            self.costList = self.getCostList(self.genePool, dataSalmonTrain, dataSeabassTrain)
            self.fitList = self.getFitList(self.costList)
            self.bestElite = self.getBestElite(self.genePool, self.fitList) # bestElite는 ErrorRate가 아니고 Fitness의 총합을 기준으로 판단
            self.bestEliteErrorRate = getErrorRate(dataSalmonTrain, dataSeabassTrain, self.bestElite)*100.0
            
            
            # 지금까지 학습 중 최상의 결과(에러율이 가장 낮은 파라메터)를 저장
            if(self.bestEliteErrorRate <= self.finalEliteRate):
                self.finalTrainCnt = copy.deepcopy(trainCnt)
                self.finalElite = copy.deepcopy(self.bestElite)
                self.finalEliteRate = copy.deepcopy(self.bestEliteErrorRate)
            
            
            # 학습결과 로그저장 및 콘솔출력    
            trainLogSentence = '%4d-th training, BestElite\'s ErrorRate: ' % trainCnt + str(self.bestEliteErrorRate) + '%'
            fdTrainLog.write(trainLogSentence + '\n')
            print trainLogSentence
            
            
            ## 정지조건 1: 적어도 300번째 이상 학습결과에서, 에러율이 9% 이하인 파라메터가 나온 경우 해당 세대를 최종결과로 선택
            ## 정지조건 2: 300번째~2000번째 학습에서 9% 이하의 파라메터를 찾지 못한 경우에는, 전체 학습결과 중 에러율이 가장 낮았던 결과를 선택
            stopCond = ( (trainCnt >= 300) and (self.bestEliteErrorRate <= 9.0) ) or ( trainCnt >=2000 )
        #while 문 종료
        
        
        # 정지조건 1에 의해 2000번 이전에 종료되었을 경우 종료된 시점의 유전자풀을 최종 결과로 사용
        if trainCnt < 2000:
            self.finalTrainCnt = copy.deepcopy(trainCnt)
            self.finalEliteRate = copy.deepcopy(self.bestEliteErrorRate)
            self.finalElite = copy.deepcopy(self.bestElite)
        
        # 학습결과 로그파일 완성 및 생성결과 콘솔 출력
        finalMsg = '\nFinal Result (%d-th generation): %.1f%% -> '  % (self.finalTrainCnt, self.finalEliteRate) + str(self.finalElite) + '\n'
        fdTrainLog.write(finalMsg + '\n')
        print finalMsg, '\nTraining result file: ' + trResFn + '\n\n'
        fdTrainLog.close()        
        
        
        
    ##### init 함수에서 이용하는 함수들 정의
    
    # 비용(cost)을 구해주는 함수
    def getCost(self, feature, param, type):     
        cost = param[0] * feature[0] + param[1] * feature[1] + param[2]
        if cost <= 0:
            if type == 'salmon':
                return 0
            else:
                return abs(cost)
        else:
            if type == 'seabass':
                return 0
            else:
                return abs(cost)
                
                
    # 유전자풀에 있는 각 파라미터들의 비용을 구해 리스트로 만들어 리턴
    def getCostList(self, genePool, dataSalmon, dataSeabass):
        costList = []
        for param in genePool:
            costSum = 0
            
            # salmon 데이터들의 각 비용 계산
            for data in dataSalmon:
                costSum += self.getCost(data, param, 'salmon')
                    
            # seabass 데이터들의 각 비용 계산
            for data in dataSeabass:
                costSum += self.getCost(data, param, 'seabass')
            
            # 데이터들의 cost의 평균을 구해서 저장
            costSum /= len(dataSalmon) + len(dataSeabass)
            costList.append(costSum)       
        return costList            
                
                
    # costList에 자연로그와 역수를 취해서 fitList 만들어 반환
    def getFitList(self, costList):
        fitList = []
        for cost in costList:
            fitList.append( 1000.0 / np.log(cost) )
        return fitList
            
                
    # fitList를 받아 다트를 던지고, 선택된 원소의 인덱스를 반환
    def selection(self, fitList):
        totalFit = sum(fitList)
        dart = ra.uniform(0.0, totalFit)
    
        now = 0.0
        selectedIndex = None
    
        for i in range(len(fitList)):
            fit = fitList[i]
            now += fit
        
            if dart <= now:
                selectedIndex = i
                break
            
        return selectedIndex
        
    
    # 유전자풀과 부모 인덱스를 받아서 Uniform Crossover된 자식을 생성해 리턴
    def makeOffspring(self, genePool, papaIndex, mamaIndex):
        offspring = []
    
        for i in range(len(genePool[papaIndex])):
            dart = ra.uniform(0.0, 1.0)
            if dart >= 0.5:
                offspring.append(genePool[papaIndex][i])
            else:
                offspring.append(genePool[mamaIndex][i])
        
        return offspring
        
        
    
    # 유전자를 받아서 확률에 따라 돌연변이로 바꾸어줌
    def mutation(self, gene, mutProb):
        newGene = []
    
        for i in range(len(gene)):
            dart = ra.uniform(0.0, 1.0)
        
            # 돌연변이가 될 확률에 걸리면(돌연변이 발생된 유전자)
            if dart <= mutProb:
                rangeMin, rangeMax = self.mutationRangeList[i]
                newGene.append(gene[i] + ra.uniform(rangeMin, rangeMax))
            
            # 돌연변이가 될 확률에 걸리지 않으면 (정상 유전된 유전자)
            else:
                newGene.append(gene[i])
    
        return newGene
        
        
      
    # 유전자풀에 있는 파라매터 중 fitness가 가장 높은 파라매터를 찾아 리턴
    def getBestElite(self, genePool, fitList):
        sortedIndexList = np.argsort(fitList)
        return genePool[sortedIndexList[-1]]
        

        

##### 프로그램 메인함수    
if __name__ == '__main__':
    argmentNum = len(sys.argv)
    
    if argmentNum == 4:  #command line argument 수 확인
        popSize = int(sys.argv[1])    # 전체 개체 수
        eliteNum = int(sys.argv[2])   # elite 개체 수
        mutProb = float(sys.argv[3])  # mutation 확률
        
        # GA 객체 생성해서 학습진행
        trainingData = GA(popSize, eliteNum, mutProb)
        
        # Test 데이터를 읽어들임
        dataSalmonTest = readFloatTxt('salmon_test.txt')
        dataSeabassTest = readFloatTxt('seabass_test.txt')
        
        # 분류결과를 저장할 로그파일 생성
        tsResFn = 'test_output_%d_%d_%.2f.txt' % (popSize, eliteNum, mutProb)
        fdTestLog = open(tsResFn, 'w') 
        print '-----Testing start-----'
        
        # salmon 데이터 분류
        for data in dataSalmonTest:
            data.append(getClass(data, trainingData.finalElite))
        
            if data[2] == 'salmon':
                data.append('correct')
            else:
                data.append('wrong')
        
            testResultMsg = 'body: %.1f tail: %.1f ==> %s (%s)' % (data[0], data[1], data[2], data[3])
            fdTestLog.write(testResultMsg + '\n')
            print testResultMsg
        
        # seabass 데이터 분류
        for data in dataSeabassTest:
            data.append(getClass(data, trainingData.finalElite))
        
            if data[2] == 'seabass':
                data.append('correct')
            else:
                data.append('wrong')
        
            testResultMsg = 'body: %.1f tail: %.1f ==> %s (%s)' % (data[0], data[1], data[2], data[3])
            fdTestLog.write(testResultMsg + '\n')
            print testResultMsg
        
        #최종 테스트결과 정확도 로그저장 및 콘솔출력
        finalResultErrorRate = getErrorRate(dataSalmonTest, dataSeabassTest, trainingData.finalElite)*100.0
        finalResultMsg = '\nTest data ErrorRate (with Final Elite): %.1f' % (finalResultErrorRate) + '%\n'
        fdTestLog.write(finalResultMsg + '\n')
        print finalResultMsg, '\nTesting result file: ' + tsResFn
        fdTestLog.close()
       
        
    else:
        print 'Usage: %s [populationSize] [eliteNum] [mutationProb]' % (op.basename(sys.argv[0]))
          
        
        