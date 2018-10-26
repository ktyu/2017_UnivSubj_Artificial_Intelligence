# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as nr
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku

nr.seed(12345)  # random seed 설정

# 데이터 정의
trFeatArr = np.array([0.0, 0.0,
                      0.0, 1.0,
                      1.0, 0.0,
                      1.0, 1.0]).reshape(-1, 2)
print 'trFeatArr', trFeatArr.shape
print trFeatArr

# 레이블 정의(one-hot representation)
trLabelArr = np.array([1.0, 0.0,
                       0.0, 1.0,
                       0.0, 1.0,
                       1.0, 0.0]).reshape(-1, 2)
print 'trLabelArr', trLabelArr.shape
print trLabelArr

# 모델 구성(2(input) -> 5(relu) -> 2(sigmoid))
model = km.Sequential()
model.add(kl.Dense(input_dim=2, units=5))
model.add(kl.BatchNormalization())
model.add(kl.Activation('relu'))
model.add(kl.Dense(units=2))
model.add(kl.BatchNormalization())
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

# 모델 파라미터 설정
model.set_weights(params)

# 테스트
res = model.predict(trFeatArr, batch_size=4)
print 'res', res.shape
print res
print np.argmax(res, axis=1)
