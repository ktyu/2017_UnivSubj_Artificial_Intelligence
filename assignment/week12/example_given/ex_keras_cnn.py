# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as nr
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku

nr.seed(12345)  # random seed 설정

# 데이터 정의
data1 = np.array([1.0, 1.0, 1.0, 0.9,
                  1.0, 0.0, 0.0, 1.0,
                  1.0, 0.0, 0.0, 1.0,
                  0.9, 1.0, 1.0, 0.9]).reshape(1, 4, 4, 1) # "0"
data2 = np.array([0.0, 1.0, 1.0, 0.0,
                  0.0, 0.9, 1.0, 0.0,
                  0.0, 1.0, 0.9, 0.0,
                  0.0, 1.0, 1.0, 0.0]).reshape(1, 4, 4, 1) # "1"
data3 = np.array([0.0, 1.0, 1.0, 0.0,
                  0.0, 0.9, 0.9, 0.0,
                  0.0, 0.9, 0.9, 0.0,
                  0.0, 0.9, 0.9, 0.0]).reshape(1, 4, 4, 1) # "1"
data4 = np.array([1.0, 0.9, 1.0, 1.0,
                  1.0, 0.0, 0.0, 1.0,
                  1.0, 0.0, 0.0, 1.0,
                  1.0, 0.9, 0.9, 1.0]).reshape(1, 4, 4, 1) # "0"
trFeatArr = np.vstack((data1, data2, data3, data4))
print 'trFeatArr', trFeatArr.shape
print trFeatArr

# 레이블 정의(one-hot representation)
trLabelArr = np.array([1.0, 0.0,
                       0.0, 1.0,
                       0.0, 1.0,
                       1.0, 0.0]).reshape(-1, 2)
print 'trLabelArr', trLabelArr.shape
print trLabelArr

# 모델 구성(4x4x1(input) -> CONV(ReLU) -> CONV(ReLU) -> FC(sigmoid)) #sp.82 참고
model = km.Sequential()
model.add(kl.Conv2D(input_shape=(4, 4, 1), filters=5,
                    kernel_size=(3, 3), strides=1,
                    padding='same'))    # zero-padding
model.add(kl.Activation('relu'))
model.add(kl.Conv2D(filters=3,
                    kernel_size=(3, 3), strides=1))
model.add(kl.Activation('relu'))
model.add(kl.Flatten()) # 하나의 벡터로 늘어놓는 작업
model.add(kl.Dense(units=2))
model.add(kl.Activation('sigmoid'))

# 학습 설정(MSE / SGD / learning rate decay / momentum)
model.compile(loss='mean_squared_error',
              optimizer=ko.SGD(lr=0.1, decay=0.01, momentum=0.9))

# 모델 구조 그리기
ku.plot_model(model, 'model.png')

# 학습(500회 반복, 4개 샘플씩 배치 학습)
model.fit(trFeatArr, trLabelArr, epochs=500, batch_size=4)

# 모델 파라미터 얻기
params = model.get_weights()
print 'params', len(params)
for i in range(len(params)):
    param = params[i]
    print i + 1, param.shape
    print param
    
'''
첫번째 파라매터 3x3x1x5 -> 3x3 이미지, 1개채널, 필터5개
'''

# 모델 파라미터 설정
model.set_weights(params)

# 테스트
res = model.predict(trFeatArr, batch_size=4)
print 'res', res.shape
print res
print np.argmax(res, axis=1)
