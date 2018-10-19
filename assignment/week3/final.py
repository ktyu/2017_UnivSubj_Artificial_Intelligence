#-*- coding: utf-8 -*-
import sys
import time
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

##### 실습과제 1번 코드 시작

try: # 입력 파일 열기
    fdSalmonTrain = open('salmon_train.txt', 'r')
    fdSeabassTrain = open('seabass_train.txt', 'r')

except: # 입력 파일 열기 중 에러 발생시 예외처리해서 종료
    print "File open failed!\nProgram will be end."
    time.sleep(5)
    sys.exit(1)
    

# 입력파일 내용을 읽어들임
dataSalmonTrain = fdSalmonTrain.readlines()
dataSeabassTrain = fdSeabassTrain.readlines()
fdSalmonTrain.close()
fdSeabassTrain.close()

# 입력된 데이터 리스트로 정리
for i in range(len(dataSalmonTrain)): 
    dataSalmonTrain[i] = dataSalmonTrain[i].strip('\r\n').split('\t')
    dataSalmonTrain[i] = map(float, dataSalmonTrain[i])
    if len(dataSalmonTrain[i]) != 2:
        del dataSalmonTrain[i]

for i in range(len(dataSeabassTrain)): 
    dataSeabassTrain[i] = dataSeabassTrain[i].strip('\r\n').split('\t')
    dataSeabassTrain[i] = map(float, dataSeabassTrain[i])
    if len(dataSeabassTrain[i]) != 2:
        del dataSeabassTrain[i]




# 파라미터를 기준으로 어종을 분류해주는 함수
def getClass(feature, param):
	res = param[0] * feature[0] + param[1] * feature[1] + param[2]

	if res < 0:
		return "salmon"
	return "seabass"
   
   
# 오류율을 계산해주는 함수   
def getError(data1, data2, param): #실제 메인함수에서 콜 -> getError(dataSalmonTrain, dataSeabassTrain, param)
    totalCnt = 0.0
    errorCnt = 0.0
    
    for data in data1:
        if getClass(data, param) == "salmon":
            totalCnt += 1.0
        else:
            totalCnt += 1.0
            errorCnt += 1.0
    
    for data in data2:
        if getClass(data, param) == "seabass":
            totalCnt += 1.0
        else:
            totalCnt += 1.0
            errorCnt += 1.0
    
    return errorCnt / totalCnt
    
    
# 온도 T초기화
t = 100.0

# 현재 위치 param 초기화
param = [2.0, -1.0, -180.0]

# 현재 위치의 오류 E계산
error = getError(dataSalmonTrain, dataSeabassTrain, param)

# 기타 필요한 변수들 선언
trainingCnt = 0
bestErrorRate = 1.0

try: #출력파일 생성
    fdTrainLog = open('train_log.txt', 'w') 
except: #파일 생성 중 에러 발생시 예외처리해서 종료
    print "File write failed!\nProgram will be end."
    time.sleep(5)
    sys.exit(1)

# 학습 시작
while t >= 0.001:
    trainingCnt += 1
    
    # 인접한 새 탐색지점 랜덤선택
    newParam = copy.deepcopy(param)
    newParam[0] += random.uniform(-0.01, 0.01)
    newParam[1] += random.uniform(-0.01, 0.01)
    newParam[2] += random.uniform(-10.0, 10.0)
    
    # 새로운 위치에서의 오류율과 기존 위치와의 오류율 차이 계산
    newError = getError(dataSalmonTrain, dataSeabassTrain, param)
    errorGap = newError - error
    
    # 기존 에러율이 크거나 같을 경우
    if errorGap <= 0:
        #새로운 위치로 이동후 학습 결과 출력
        param = copy.deepcopy(newParam)
        error = copy.deepcopy(newError)
        print "%4d th training -> Move Low  / ErrorRate: %.2f%% / ErrorGap: %.2f / T: %f" % (trainingCnt, error*100, errorGap, t)
        fdTrainLog.write("%4d th training -> Move Low  / ErrorRate: %.2f%% / ErrorGap: %.2f / T: %f\n" % (trainingCnt, error*100, errorGap, t))
        
        
    # 새로운 위치의 에러율이 클 경우    
    else:
        #랜덤값 r 생성
        r = random.uniform(0,1)
        
        # 온도와 r값에 따라 이동 여부 판단
        if r < np.exp(-errorGap / t):
            #이동할 경우, 새로운 학습결과 저장하고 출력
            param = copy.deepcopy(newParam)
            error = copy.deepcopy(newError)
            print "%4d th training -> Move high / ErrorRate: %.2f%% / ErrorGap: %.2f / T: %f" % (trainingCnt, error*100, errorGap, t)
            fdTrainLog.write("%4d th training -> Move high / ErrorRate: %.2f%% / ErrorGap: %.2f / T: %f\n" % (trainingCnt, error*100, errorGap, t))
            
        else:
            #이동하지 않고 머무를 경우, 결과만 출력
            print "%4d th training -> Stay here / ErrorRate: %.2f%% / ErrorGap: %.2f / T: %f" % (trainingCnt, error*100, errorGap, t)
            fdTrainLog.write("%4d th training -> Stay here / ErrorRate: %.2f%% / ErrorGap: %.2f / T: %f\n" % (trainingCnt, error*100, errorGap, t))
        

    # 최상의 결과를 저장하기 위해 기존 학습결과들과 비교
    if bestErrorRate > error:
        bestErrorRate = copy.deepcopy(error)
        bestTrainingIdx = copy.deepcopy(trainingCnt)
        bestParam = copy.deepcopy(param)
        
    # T 값 감소   
    t *= 0.99

# 학습 종료 후 최적의 결과 출력
print "\nBest paramiter: " + str(bestParam) + " in %dth training" % bestTrainingIdx
print "Best Error rate: " + str(bestErrorRate*100) + "%"
print "Total Trained time: " + str(trainingCnt)

fdTrainLog.write("\nBest paramiter: " + str(bestParam) + " in %dth training\n" % bestTrainingIdx)
fdTrainLog.write("Best Error rate: " + str(bestErrorRate*100) + "%\n")
fdTrainLog.write("Total Trained time: " + str(trainingCnt) + "\n")

fdTrainLog.close()
print "\n\"train_log.txt\" file is created!"



##### 실습과제 2번 코드 시작   

try: # 입력 파일 열기
    fdSalmonTest = open('salmon_test.txt', 'r')
    fdSeabassTest = open('seabass_test.txt', 'r')

except: # 입력 파일 열기 중 에러 발생시 예외처리해서 종료
    print "File open failed!\nProgram will be end."
    time.sleep(5)
    sys.exit(1)
    

# 입력파일 내용을 읽어들임
dataSalmonTest = fdSalmonTest.readlines()
dataSeabassTest = fdSeabassTest.readlines()
fdSalmonTest.close()
fdSeabassTest.close()

# 입력된 데이터 리스트로 정리
for i in range(len(dataSalmonTest)): 
    dataSalmonTest[i] = dataSalmonTest[i].strip('\r\n').split('\t')
    dataSalmonTest[i] = map(float, dataSalmonTest[i])
    if len(dataSalmonTest[i]) != 2:
        del dataSalmonTest[i]

for i in range(len(dataSeabassTest)): 
    dataSeabassTest[i] = dataSeabassTest[i].strip('\r\n').split('\t')
    dataSeabassTest[i] = map(float, dataSeabassTest[i])
    if len(dataSeabassTest[i]) != 2:
        del dataSeabassTest[i]

# Test 데이터 출력 준비
print "\n\n***** Result of test data *****"
testCorrectCnt = 0
testWrongCnt = 0

try: #출력파일 생성
    fdTestOutput = open('test_output.txt', 'w') 
except: #파일 생성 중 에러 발생시 예외처리해서 종료
    print "File write failed!\nProgram will be end."
    time.sleep(5)
    sys.exit(1)


# Test 데이터를 위에서 학습한 선형 분류기(가장 좋았던 파라미터)로 분류
for data in dataSalmonTest:
    data.append(getClass(data, bestParam))
    if data[2] == "salmon":
        data.append("correct")
        testCorrectCnt += 1
    else:
        data.append("wrong")
        testWrongCnt += 1
    print "body: %.1f tail: %.1f ==> %s (%s)" % (data[0], data[1], data[2], data[3])
    fdTestOutput.write("body: %.1f tail: %.1f ==> %s (%s)" % (data[0], data[1], data[2], data[3]) + "\n")

for data in dataSeabassTest:
    data.append(getClass(data, bestParam))
    if data[2] == "seabass":
        data.append("correct")
        testCorrectCnt += 1
    else:
        data.append("wrong")
        testWrongCnt += 1
    print "body: %.1f tail: %.1f ==> %s (%s)" % (data[0], data[1], data[2], data[3])
    fdTestOutput.write("body: %.1f tail: %.1f ==> %s (%s)" % (data[0], data[1], data[2], data[3]) + "\n")

# Test 데이터 분류 결과 저장 및 출력    
print "\nTest data Error rate: " + str(testWrongCnt*100.0 / (testCorrectCnt + testWrongCnt)) + "%"
fdTestOutput.write("\nTest data Error rate: " + str(testWrongCnt*100.0 / (testCorrectCnt + testWrongCnt)) + "%\n")
fdTestOutput.close()
print "\n\"test_output.txt\" file is created!"    



# 이미지 출력을 위해 맞은것/틀린것 분류
salmonTestCorrect = []
salmonTestWrong = []

for data in dataSalmonTest:
    if data[3] == "correct":
        salmonTestCorrect.append(data)
    else:
        salmonTestWrong.append(data)

seabassTestCorrect  = []
seabassTestWrong = []
for data in dataSeabassTest:
    if data[3] == "correct":
        seabassTestCorrect.append(data)
    else:
        seabassTestWrong.append(data)

        
# Matplotlib 을 이용한 이미지 출력
fig, ax = plt.subplots()

xList = []
yList = []
for data in salmonTestCorrect:
    x, y = data[0:2]
    xList.append(x)
    yList.append(y)
ax.plot(xList, yList, 'b^', label='salmon_test_correct')

xList = []
yList = []
for data in salmonTestWrong:
    x, y = data[0:2]
    xList.append(x)
    yList.append(y)
ax.plot(xList, yList, 'r^', label='salmon_test_wrong')

xList = []
yList = []
for data in seabassTestCorrect:
    x, y = data[0:2]
    xList.append(x)
    yList.append(y)
ax.plot(xList, yList, 'bs', label='seabass_test_correct')

xList = []
yList = []
for data in seabassTestWrong:
    x, y = data[0:2]
    xList.append(x)
    yList.append(y)
ax.plot(xList, yList, 'rs', label='seabass_test_wrong')


ax.grid(True) # 격자무늬 출력
ax.legend(loc='upper right') # 범례 위치
ax.set_xlabel('length of body') # x축 이름
ax.set_ylabel('length of tail') # y축 이름
ax.set_xlim((None, None)) # x축 값 범위
ax.set_ylim((None, None)) # y축 값 범위
plt.savefig('test_output.png') # 이미지 파일 저장
plt.show() # 그래프 출력
