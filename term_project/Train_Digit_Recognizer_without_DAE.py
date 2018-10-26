# -*- coding: utf-8 -*-

'''
    특이 사항
        1. module_load_data.py 모듈을 import 하기 때문에 파일 경로에 한글이 포함되어 있으면 에러가 납니다.
'''

import module_load_data
import os
import os.path as op
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

# 출력 이미지 경로
_DST_PATH = u'.\\images\\Train_Digit_Recognizer_without_DAE'


# 프로그램 메인함수    
if __name__ == '__main__':

    # random seed 설정
    nr.seed(12345)
    
    # 학습 데이터 로드 (노이즈가 없는 정상 데이터)
    pure_trDataList, trLabelList = module_load_data.loadMNIST_pure_trainData()
    print 'MNIST Data load done.\n'
    
    
    # 모델 구성(28x28x1(input) -> Flatten -> FC(BN, ReLU) -> FC(BN, ReLU) -> FC(BN, sigmoid))
    inputFeat = kl.Input(shape=(_N_ROW, _N_COL, 1))
    
    flatten = kl.Flatten()(inputFeat)
    
    dense1 = kl.Dense(units=100)(flatten)
    bn1 = kl.BatchNormalization()(dense1)
    relu1 = kl.Activation('relu')(bn1)
    
    dense2 = kl.Dense(units=30)(relu1)
    bn2 = kl.BatchNormalization()(dense2)
    relu2 = kl.Activation('relu')(bn2)

    dense3 = kl.Dense(units=_N_CLASS)(relu2)
    bn3 = kl.BatchNormalization()(dense3)
    output = kl.Activation('sigmoid')(bn3)
    
    modelFull = km.Model(inputs=[inputFeat], outputs=[output])

    # 학습 설정(MSE / SGD / learning rate decay / momentum)
    modelFull.compile(loss='mean_squared_error', optimizer=ko.SGD(lr=0.1, decay=0.001, momentum=0.9))
    
    # 학습 진행 (10 Epochs)
    modelFull.fit(pure_trDataList, trLabelList, epochs=10, batch_size=100)
    
    # 모델 구조 이미지를 저장할 경로 생성
    if op.exists(_DST_PATH) == False:
        os.mkdir(_DST_PATH)
        
    if op.exists(_DST_PATH + u'\\model') == False:
        os.mkdir(_DST_PATH + u'\\model')
    
    # 모델 구조 그리기
    ku.plot_model(modelFull, _DST_PATH + u'\\model\\Train_Digit_Recognizer_without_DAE.png') 
    print "\nA Image is saved in 'images\\Train_Digit_Recognizer_without_DAE' folder."
    
    # 모델 파라미터를 파일로 저장
    km.save_model(modelFull, 'Train_Digit_Recognizer_without_DAE_param.h5')
    print "'Train_Digit_Recognizer_without_DAE_param.h5' file is saved."
