# -*- coding: utf-8 -*-

'''
    특이 사항
        1. module_load_data.py 모듈을 import 하기 때문에 파일 경로에 한글이 포함되어 있으면 에러가 납니다.
'''

import module_load_data
import numpy as np
import numpy.random as nr
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku


# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_CLASS = 10


# 프로그램 메인함수    
if __name__ == '__main__':
    
    # random seed 설정
    nr.seed(12345)
    
    # 3가지 종류의 노이즈가 섞여있는 NMIST Data Set의 테스트 데이터 로드 
    nmnist_AWGN_tsDataList, nmnist_AWGN_tsLabelList = module_load_data.loadNMNIST_AWGN_testData() # AWGN(Additive White Gaussian Noise)
    nmnist_MB_tsDataList, nmnist_MB_tsLabelList = module_load_data.loadNMNIST_MB_testData() # Motion Blur
    nmnist_RC_tsDataList, nmnist_RC_tsLabelList = module_load_data.loadNMNIST_RC_testData() # reduced contrast and AWGN
    print 'N-MNIST Data load done.\n'

    
    ### DAE 없이 학습된 숫자 인식기 모델 파라미터 로드
    print '---Test without DAE---'
    modelFull = km.load_model('Train_Digit_Recognizer_without_DAE_param.h5')
    
    ## AWGN(Additive White Gaussian Noise) Data Set 인식
    # 숫자 인식기 테스트 진행
    res = modelFull.predict(nmnist_AWGN_tsDataList, batch_size=100)
    
    # 결과 계산
    classificationResult = np.argmax(res, axis=1)
    answerLabel = np.argmax(nmnist_AWGN_tsLabelList, axis=1)
    
    # 모든 샘플에 대한 최종 오류율 계산 및 출력
    testResult = (classificationResult == answerLabel).astype('float32')
    testResultErrorRate = ((testResult.shape[0]-np.sum(testResult)) / testResult.shape[0]) * 100
    print 'AWGN -> Total Error Rate: %.2f%% (%5d / %5d)' % (testResultErrorRate, testResult.shape[0]-np.sum(testResult), testResult.shape[0])
    
    ## Motion Blur Data Set 인식
    # 숫자 인식기 테스트 진행
    res = modelFull.predict(nmnist_MB_tsDataList, batch_size=100)
    
    # 결과 계산
    classificationResult = np.argmax(res, axis=1)
    answerLabel = np.argmax(nmnist_MB_tsLabelList, axis=1)
    
    # 모든 샘플에 대한 최종 오류율 계산 및 출력
    testResult = (classificationResult == answerLabel).astype('float32')
    testResultErrorRate = ((testResult.shape[0]-np.sum(testResult)) / testResult.shape[0]) * 100
    print ' MB  -> Total Error Rate: %.2f%% (%5d / %5d)' % (testResultErrorRate, testResult.shape[0]-np.sum(testResult), testResult.shape[0])
    
    
    ## reduced contrast and AWGN Data Set 인식
    # 숫자 인식기 테스트 진행
    res = modelFull.predict(nmnist_RC_tsDataList, batch_size=100)
    
    # 결과 계산
    classificationResult = np.argmax(res, axis=1)
    answerLabel = np.argmax(nmnist_RC_tsLabelList, axis=1)
    
    # 모든 샘플에 대한 최종 오류율 계산 및 출력
    testResult = (classificationResult == answerLabel).astype('float32')
    testResultErrorRate = ((testResult.shape[0]-np.sum(testResult)) / testResult.shape[0]) * 100
    print ' RC  -> Total Error Rate: %.2f%% (%5d / %5d)\n\n' % (testResultErrorRate, testResult.shape[0]-np.sum(testResult), testResult.shape[0])
    
    
#---------------------------------------------------------------------


    # DAE 모델 파라미터 로드
    denosingAutoEncoder = km.load_model('DAE_model_param.h5')
    print 'Denoising Auto Encoder Model is loaded.'
    
    # Test Data들을 Denoising 시키기 위해 DAE를 통과시킴
    nmnist_AWGN_tsDataList = denosingAutoEncoder.predict(nmnist_AWGN_tsDataList)
    print 'AWGN(Additive White Gaussian Noise) Data Set Denoising Done.'
    
    nmnist_MB_tsDataList = denosingAutoEncoder.predict(nmnist_MB_tsDataList)
    print 'Motion Blur Data Set Denoising Done.'

    nmnist_RC_tsDataList = denosingAutoEncoder.predict(nmnist_RC_tsDataList)
    print 'Reduced Contrast and AWGN Data Set Denoising Done.\n'    
    
    # DAE를 포함해 학습한 숫자 인식기 모델 파라미터 로드
    print '---Test with DAE---'
    modelFull = km.load_model('Train_Digit_Recognizer_param.h5')
    
    ## AWGN(Additive White Gaussian Noise) Data Set 인식
    # 숫자 인식기 테스트 진행
    res = modelFull.predict(nmnist_AWGN_tsDataList, batch_size=100)
    
    # 결과 계산
    classificationResult = np.argmax(res, axis=1)
    answerLabel = np.argmax(nmnist_AWGN_tsLabelList, axis=1)
    
    # 모든 샘플에 대한 최종 오류율 계산 및 출력
    testResult = (classificationResult == answerLabel).astype('float32')
    testResultErrorRate = ((testResult.shape[0]-np.sum(testResult)) / testResult.shape[0]) * 100
    print 'AWGN -> Total Error Rate: %.2f%% (%5d / %5d)' % (testResultErrorRate, testResult.shape[0]-np.sum(testResult), testResult.shape[0])
    
    
    ## Motion Blur Data Set 인식
    # 숫자 인식기 테스트 진행
    res = modelFull.predict(nmnist_MB_tsDataList, batch_size=100)
    
    # 결과 계산
    classificationResult = np.argmax(res, axis=1)
    answerLabel = np.argmax(nmnist_MB_tsLabelList, axis=1)
    
    # 모든 샘플에 대한 최종 오류율 계산 및 출력
    testResult = (classificationResult == answerLabel).astype('float32')
    testResultErrorRate = ((testResult.shape[0]-np.sum(testResult)) / testResult.shape[0]) * 100
    print ' MB  -> Total Error Rate: %.2f%% (%5d / %5d)' % (testResultErrorRate, testResult.shape[0]-np.sum(testResult), testResult.shape[0])
    
    
    ## reduced contrast and AWGN Data Set 인식
    # 숫자 인식기 테스트 진행
    res = modelFull.predict(nmnist_RC_tsDataList, batch_size=100)
    
    # 결과 계산
    classificationResult = np.argmax(res, axis=1)
    answerLabel = np.argmax(nmnist_RC_tsLabelList, axis=1)
    
    # 모든 샘플에 대한 최종 오류율 계산 및 출력
    testResult = (classificationResult == answerLabel).astype('float32')
    testResultErrorRate = ((testResult.shape[0]-np.sum(testResult)) / testResult.shape[0]) * 100
    print ' RC  -> Total Error Rate: %.2f%% (%5d / %5d)' % (testResultErrorRate, testResult.shape[0]-np.sum(testResult), testResult.shape[0])