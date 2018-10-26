# -*- coding: utf-8 -*-
import numpy.random as nr
import keras.models as km
import keras.layers as kl



def printWeight(model):
    # 모델 파라미터 얻기
    params = model.get_weights()
    print 'params', len(params)
    for i in range(len(params)):
        param = params[i]
        print i + 1, param.shape
        print param
    print
    


if __name__ == '__main__':
    nr.seed(12345)  # random seed 설정
    
    # 모델 구성
    model = km.Sequential()
    
    print 'add FC'
    model.add(kl.Dense(input_dim=2, units=5))
    printWeight(model)
    
    print 'add BN'
    model.add(kl.BatchNormalization())
    printWeight(model)
    
    print 'add ReLU'
    model.add(kl.Activation('relu'))
    printWeight(model)
    
    print 'add FC'
    model.add(kl.Dense(units=2))
    printWeight(model)
    
    print 'add BN'
    model.add(kl.BatchNormalization())
    printWeight(model)
    
    print 'add sigmoid'
    model.add(kl.Activation('sigmoid'))
    printWeight(model)
    