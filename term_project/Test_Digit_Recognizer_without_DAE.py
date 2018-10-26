# -*- coding: utf-8 -*-

'''
    특이 사항
        1. module_load_data.py 모듈을 import 하기 때문에 파일 경로에 한글이 포함되어 있으면 에러가 납니다.
        2. 첫번째 인자로 테스트시 데이터에 노이즈를 섞을 비율을 입력해주면 됩니다. (0.0 ~ 1.0)
'''

import module_load_data
import numpy as np
import numpy.random as nr
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku
import sys


# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_CLASS = 10


# 프로그램 메인함수    
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print 'Please enter 1 argument as noiseRate! (0.0~1.0)'
        sys.exit(1)

    noiseRate = float(sys.argv[1])
    pureRate = 1.0 - noiseRate
            
    # random seed 설정
    nr.seed(12345)
    
    # 테스트 데이터 로드 (정상 데이터, 노이즈가 섞인 데이터)
    pure_tsDataList, tsLabelList = module_load_data.loadMNIST_pure_testData()
    noise_tsDataList, tsLabelList = module_load_data.loadMNIST_noise_testData()

    print 'MNIST Data load done.'
    
    # Pure 데이터와 Noise 데이터를 70:30으로 섞기위해 인덱싱
    indexList = np.arange(pure_tsDataList.shape[0]) # 10000개의 인덱스 번호 생성
    nr.shuffle(indexList)
        
    pure_tsDataList_sliced = pure_tsDataList[indexList[0 : int(indexList.shape[0]*pureRate)]]
    noise_tsDataList_sliced = noise_tsDataList[indexList[int(indexList.shape[0]*pureRate) : indexList.shape[0]]]
    
    mixed_tsDataList = np.vstack((pure_tsDataList_sliced, noise_tsDataList_sliced))
    mixed_tsLabelList = tsLabelList[indexList]
    
    print 'Pure and Noise Data mixing done. (%d:%d)\n' % (pureRate*100, noiseRate*100)
    
    # 숫자 인식기 모델 파라미터 로드
    modelFull = km.load_model('Train_Digit_Recognizer_without_DAE_param.h5')
    
    # 숫자 인식기 테스트 진행
    res = modelFull.predict(mixed_tsDataList, batch_size=100)
    
    # 결과 계산
    classificationResult = np.argmax(res, axis=1)
    answerLabel = np.argmax(mixed_tsLabelList, axis=1)
    
    # 모든 샘플에 대한 최종 오류율 계산 및 출력
    testResult = (classificationResult == answerLabel).astype('float32')
    testResultErrorRate = ((testResult.shape[0]-np.sum(testResult)) / testResult.shape[0]) * 100
    
    print 'Total Error Rate: %.2f%% (%5d / %5d)' % (testResultErrorRate, testResult.shape[0]-np.sum(testResult), testResult.shape[0])