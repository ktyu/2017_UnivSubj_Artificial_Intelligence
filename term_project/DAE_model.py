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


# 출력 이미지 경로
_DST_PATH = u'.\\images\\DAE_model'


# 프로그램 메인함수    
if __name__ == '__main__':
   
    # random seed 설정
    nr.seed(12345)

    # 학습 데이터 로드 (정상 데이터, 노이즈가 섞인 데이터)
    pure_trDataList, trLabelList = module_load_data.loadMNIST_pure_trainData()
    noise_trDataList, trLabelList = module_load_data.loadMNIST_noise_trainData()
    
    # 테스트 데이터 로드 (정상 데이터, 노이즈가 섞인 데이터)
    pure_tsDataList, tsLabelList = module_load_data.loadMNIST_pure_testData()
    noise_tsDataList, tsLabelList = module_load_data.loadMNIST_noise_testData()

    print 'MNIST Data load done.\n'
    
    
    
    # DAE 학습을 위한 모델 생성
    input = kl.Input(shape=(_N_ROW, _N_COL, 1))

    encode_conv1 = kl.Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(input)
    enconde_max1 = kl.MaxPooling2D(pool_size=(2, 2), padding='same')(encode_conv1)
    encode_conv2 = kl.Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(enconde_max1)
    encoded = kl.MaxPooling2D(pool_size=(2, 2), padding='same')(encode_conv2)

    # Encoder의 구성은 여기까지, 샘플의 크기는 (7x7x32)

    decode_conv1 = kl.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(encoded)
    decode_max1 = kl.UpSampling2D(size=(2, 2))(decode_conv1)
    decode_conv2 = kl.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(decode_max1)
    decode_max2 = kl.UpSampling2D(size=(2, 2))(decode_conv2)
    decoded = kl.Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(decode_max2)

    denosingAutoEncoder = km.Model(inputs=[input], outputs=[decoded]) # DAE 전체 모델
    encoder = km.Model(inputs=[input], outputs=[encoded]) # Enconder 부분의 모델
    
    # 학습 설정
    denosingAutoEncoder.compile(loss='binary_crossentropy', optimizer='adadelta')
    
    # 모델 구조 이미지를 저장할 경로 생성
    if op.exists(_DST_PATH) == False:
        os.mkdir(_DST_PATH)
        
    if op.exists(_DST_PATH + u'\\model') == False:
        os.mkdir(_DST_PATH + u'\\model')
    
    # 모델 구조 그리기
    ku.plot_model(denosingAutoEncoder, _DST_PATH + u'\\model\\DAE_entire.png') 
    ku.plot_model(encoder, _DST_PATH + u'\\model\\DAE_part_encoder.png') 
    
    # trDataList로 학습 진행
    denosingAutoEncoder.fit(noise_trDataList, pure_trDataList, epochs=3, batch_size=100)
                    
    # 모델 파라미터 저장
    km.save_model(denosingAutoEncoder, 'DAE_model_param.h5') # 전체 DAE 모델만 저장
    
    
    # DAE 성능을 눈으로 확인하기 위해 tsDataList로 테스트
    noise_to_encoder = encoder.predict(noise_tsDataList)
    pure_to_encoder = encoder.predict(pure_tsDataList)
    noise_to_DAE = denosingAutoEncoder.predict(noise_tsDataList)
    pure_to_DAE = denosingAutoEncoder.predict(pure_tsDataList)

    print '\nnoise_to_encoder', noise_to_encoder.shape
    print 'pure_to_encoder', pure_to_encoder.shape
    print 'noise_to_encoder', noise_to_encoder.shape
    print 'pure_to_encoder', pure_to_encoder.shape, '\n'
    
    # 샘플들을 출력 해보기 위해 경로 생성
    if op.exists(_DST_PATH + u'\\data_samples') == False:
        os.mkdir(_DST_PATH + u'\\data_samples')
    
    # noise가 섞인 테스트 데이터의 DAE에 통과 전, Encoding만 진행한 후, Denosing 을 전부 진행한 후 결과를 각 5개씩 출력    
    for i in range(5):
        module_load_data.drawImage(noise_tsDataList[i, :, :, 0], _DST_PATH + u'\\data_samples\\noise_to_DAE_before_%d_label_%d.png' % (i, tsLabelList[i].argmax()))

    for j in range(32):
        for i in range(5):
            module_load_data.drawImage(noise_to_encoder[i, :, :, j], _DST_PATH + u'\\data_samples\\noise_to_DAE_enconded_%d_label_%d_filter_%d.png' % (i, tsLabelList[i].argmax(), j))
        
    for i in range(5):
        module_load_data.drawImage(noise_to_DAE[i, :, :, 0], _DST_PATH + u'\\data_samples\\noise_to_DAE_denoised_%d_label_%d.png' % (i, tsLabelList[i].argmax()))
    
    
    # pure 한 테스트 데이터의 DAE에 통과 전, Encoding만 진행한 후, Denosing 을 전부 진행한 후 결과를 각 5개씩 출력
    for i in range(5):
        module_load_data.drawImage(pure_tsDataList[i, :, :, 0], _DST_PATH + u'\\data_samples\\pure_to_DAE_before_%d_label_%d.png' % (i, tsLabelList[i].argmax()))
    
    for j in range(32):
        for i in range(5):
            module_load_data.drawImage(pure_to_encoder[i, :, :, j], _DST_PATH + u'\\data_samples\\pure_to_DAE_enconded_%d_label_%d_filter_%d.png' % (i, tsLabelList[i].argmax(), j))
        
    for i in range(5):
        module_load_data.drawImage(pure_to_DAE[i, :, :, 0], _DST_PATH + u'\\data_samples\\pure_to_DAE_denoised_%d_label_%d.png' % (i, tsLabelList[i].argmax()))
    
    print "Images are saved in 'images\\DAE_model' folder."
