#-*- coding: utf-8 -*-

import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt

_DIM = 2
_N_GMM = 4


# 실수 데이터가 담긴 txt파일을 인자로 받아서 numpy 행렬로 정리해서 리턴해주는 함수
def readFloatTxt(filename):
    fd = open(filename, 'r')
    data = fd.readlines()
    fd.close()
        
    for i in range(len(data)): 
        data[i] = data[i].strip('\r\n').split('\t')
        data[i] = map(float, data[i])
        
    return np.array(data).T

    
# _DIM 차원에서의 가우시안 PDF를 구해주는 함수
def getDensity(xVec, meanVec, diagVarVec):
    t1 = np.power(2.0 * np.pi, _DIM / 2.0)
    t2 = np.power(np.prod(diagVarVec), 0.5)
    
    diff = (xVec - meanVec).reshape(-1,1)
    invDiagVarVec = np.diag(1.0 / diagVarVec)
    t3 = np.dot(diff.T, invDiagVarVec)[0]
    t3 = np.dot(t3, diff)[0]
    t3 *= -0.5
    
    result = (1.0 / (t1* t2)) * np.exp(t3)

    return result
    
    
# Matplotlib 을 이용한 이미지 출력
def drawImage(xVec, meanVec, fileName):
    fig, ax = plt.subplots()

    xList = xVec[0, :]
    yList = xVec[1, :]
    ax.plot(xList, yList, 'ko', label='training samples')

    xList = meanVec[0,:]
    yList = meanVec[1,:]
    ax.plot(xList, yList, 'r*', label='component means')

    ax.grid(True) # 격자무늬 출력
    ax.legend(loc='upper right') # 범례 위치
    ax.set_xlabel('feat_1') # x축 이름
    ax.set_ylabel('feat_2') # y축 이름
    #ax.set_xlim((None, None)) # x축 값 범위
    #ax.set_ylim((None, None)) # y축 값 범위
    plt.savefig(fileName) # 이미지 파일 저장
    #plt.show() # 그래프 출력

    # flush
    plt.cla()
    plt.clf()
    plt.close()
    
    
# 메인함수    
if __name__ == '__main__':
    
    # random seed 설정
    #nr.seed(12345)  
    
    # donut.txt 데이터 로드
    xVec = readFloatTxt('donut.txt')

    # 총합이 1인 혼합 가중치 벡터 랜덤 생성
    modelWeight = nr.rand(_N_GMM)
    modelWeight /= np.sum(modelWeight)
    
    # 평균과 분산 벡터를 적정 범위에서 랜덤 생성
    meanVec = nr.rand(_DIM, _N_GMM)*20-10 # -10~10 사이 랜덤값 생성
    diagVarVec = nr.rand(_DIM, _N_GMM)*4+1 # 1~5 사이 랜덤값 생성
    
    
    # 벡터 형태 확인(출력)
    print 'xVec', xVec.shape
    print 'meanVec', meanVec.shape
    print 'diagVarVec', diagVarVec.shape
    #drawImage(xVec, meanVec, 'output_0.png')
    

    cnt = 0
    
    # EM 알고리즘을 반복
    while(cnt < 50):
    
        # E 단계 (모든 샘플들에 대해 반복)
        expectaion = []
        for i in range(xVec.shape[1]):
            sample = []
            
            # i 번째 샘플이 j 번째 성분에 속할 확률을 계산
            for j in range(_N_GMM): 
                sample.append( modelWeight[j] * getDensity(xVec[:, i], meanVec[:, j], diagVarVec[:, j]) )
            sum = np.sum(sample)
            for j in range(_N_GMM):
                sample[j] /= sum
            
            # i 번째 샘플에 대한 계산 결과를 저장
            expectaion.append(sample)
        
        # 계산한 모든 샘플들에 대한 결과를 numpy array로 저장
        expectaionResult = np.array(expectaion)
        
        
        # M 단계
        for j in range(_N_GMM):
            n = np.sum(expectaionResult[:, j])
            
            # 평균 갱신
            meanVec[:, j] = np.sum(expectaionResult[:, j] *xVec, axis=1) / n
            
            # 분산 갱신 (대각 성분만 갱신)
            diagVarVec[:, j] = (np.sum(expectaionResult[:, j] *xVec * xVec, axis=1) / n) - np.power(meanVec[:, j], 2)
            
            # 가중치 갱신
            modelWeight[j] = n / xVec.shape[1]
        
        
        # 반복횟수 증가
        cnt += 1
        #drawImage(xVec, meanVec, 'output_%d.png' % (cnt))
        
        
    # 최종결과 이미지 출력    
    drawImage(xVec, meanVec, 'output.png')
    print "'output.png' file is saved."
    