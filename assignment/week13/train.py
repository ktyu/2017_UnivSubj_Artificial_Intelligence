# -*- coding: utf-8 -*-
import struct as st
import numpy as np
import numpy.random as nr
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku


# MNIST 데이터 경로
_SRC_PATH = u'..\\'
_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images.idx3-ubyte'
_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels.idx1-ubyte'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL

# 기타 필요한 상수 정의
_N_CLASS = 10
_BATCH_SIZE = 100  # 가급적 데이터 갯수의 약수 중 하나로
_TEST_SET_SIZE = 1000


def loadData(fn):
      
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    nRow = st.unpack('>I', fd.read(4))[0]
    nCol = st.unpack('>I', fd.read(4))[0]
    
    # data: unsigned byte
    dataList = []
    for i in range(nData):
        dataRawList = fd.read(_N_PIXEL)
        dataNumList = st.unpack('B' * _N_PIXEL, dataRawList)
        dataArr = np.array(dataNumList).reshape(_N_ROW, _N_COL, 1)
        dataList.append(dataArr.astype('float32'))
        
    fd.close()
    return np.array(dataList)


def loadLabel(fn):

    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
      
    # data: unsigned byte
    labelList = []
    for i in range(nData):
        dataLabel = st.unpack('B', fd.read(1))[0]
        label = np.zeros(_N_CLASS)
        label[dataLabel] = 1.0
        labelList.append(label)
        
    fd.close()
    return np.array(labelList)


def loadMNIST():
    # 학습 데이터 / 레이블 로드
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)
    
    return trDataList, trLabelList

        
# 프로그램 메인함수    
if __name__ == '__main__':
    
    # random seed 설정
    nr.seed(12345)  
    
    # MNIST 데이터셋 로드
    trDataList, trLabelList = loadMNIST()
    print "trDataList: ", trDataList.shape
    print "trLabelList: ", trLabelList.shape
    print "MNIST Data load done.\n"    
    
    # 1 Epoch의 배치사이즈를 조정하기 위해 인덱스를 기준으로 Validation
    indexList = np.arange(trDataList.shape[0]) # 60000개의 인덱스 번호 생성
    nr.shuffle(indexList)
    iterationNum = int(trDataList.shape[0]-_TEST_SET_SIZE) / _BATCH_SIZE # (60000-1000) / 100 = 590회 학습시 Training set 전체 1회 학습
    
    # Test Set 으로 쓸 데이터 인덱스 저장 -> 전체 Data set의 뒤에서 부터 _TEST_SET_SIZE 갯수 만큼의 데이터는 오류율 검사용도로 사용
    tsIndexList = indexList[trDataList.shape[0]-_TEST_SET_SIZE:trDataList.shape[0]] 

    # Training Set 으로 쓸 데이터 인덱스 저장 -> 나머지는 _BATCH_SIZE 갯수만큼 쪼개서 각각 저장
    trIndexList = []
    for i in range(iterationNum):
        trIndexList.append(indexList[i*_BATCH_SIZE : (i+1)*_BATCH_SIZE])

    '''
    # 모델 구성(28x28x1(input) -> CONV(ReLU) -> CONV(ReLU) -> FC(sigmoid))
    model = km.Sequential()
    
    model.add(kl.Conv2D(input_shape=(_N_ROW, _N_COL, 1), filters=5,
                        kernel_size=(3, 3), strides=1, padding='same')) # zero-padding
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    
    model.add(kl.Conv2D(filters=3,
                        kernel_size=(3, 3), strides=1))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    
    model.add(kl.Flatten()) # 하나의 벡터로 늘어놓는 작업
    
    model.add(kl.Dense(units=_N_CLASS))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('sigmoid'))
    '''
    
    # 모델 구성(28x28x1(input) -> CONV(ReLU) -> CONV(ReLU) -> FC(sigmoid))
    inputFeat = kl.Input(shape=(_N_ROW, _N_COL, 1))
    
    conv1 = kl.Conv2D(filters=5, kernel_size=(3, 3), strides=1, padding='same')(inputFeat)
    bn1 = kl.BatchNormalization()(conv1)
    relu1 = kl.Activation('relu')(bn1)
    
    conv2 = kl.Conv2D(filters=3, kernel_size=(3, 3), strides=1)(relu1)
    bn2 = kl.BatchNormalization()(conv2)
    relu2 = kl.Activation('relu')(bn2)
    
    flatten = kl.Flatten()(relu2)
    
    dense = kl.Dense(units=_N_CLASS)(flatten)
    bn3 = kl.BatchNormalization()(dense)
    output = kl.Activation('sigmoid')(bn3)
    
    modelFull = km.Model(inputs=[inputFeat], outputs=[output])
    modelRelu1 = km.Model(inputs=[inputFeat], outputs=[relu1])

    # 학습 설정(MSE / SGD / learning rate decay / momentum)
    modelFull.compile(loss='mean_squared_error', optimizer=ko.SGD(lr=0.1, decay=0.001, momentum=0.9))
    modelRelu1.compile(loss='mean_squared_error', optimizer=ko.SGD(lr=0.1, decay=0.001, momentum=0.9))
    
    #ku.plot_model(modelFull, 'modelFull.png') # 모델 구조 그리기
    #ku.plot_model(modelRelu1, 'modelRelu1.png') # 모델 구조 그리기
    
    
    print "-----Training Start-----"
    
    # 학습 준비
    trainCnt = 0
    bestErrorRate = 100.0
    fdTrainLog = open('train_log.txt', 'w') 
    
    print "Batch Size = %d\n" % (_BATCH_SIZE)
    fdTrainLog.write("Batch Size = %d\n\n" % (_BATCH_SIZE))
    
    # 학습 시작 (2000번 이상 학습하거나 Test set에 대한 에러율이 8% 미만으로 나오면 중단)
    while trainCnt < 2000:
        # Batch Size 만큼의 데이터로 1회 학습 진행 (Batch Size 만큼의 데이터로 1회)
        batch = trainCnt % iterationNum
        trLoss = modelFull.train_on_batch(trDataList[trIndexList[batch]], trLabelList[trIndexList[batch]])
        
        # Test Set 으로 에러율 검사
        result = modelFull.predict(trDataList[tsIndexList], batch_size=_BATCH_SIZE)
        testResult = (np.argmax(result, axis=1) == np.argmax(trLabelList[tsIndexList], axis=1)).astype('float32')
        errorRate = ((_TEST_SET_SIZE - np.sum(testResult)) / _TEST_SET_SIZE) * 100
        
        # 학습 결과 출력 및 로그저장
        print '%4dth Training ---> errorRate: %.2f%%, trLoss: %f' % (trainCnt+1, errorRate, trLoss)
        fdTrainLog.write('%4dth Training ---> errorRate: %.2f%%, trLoss: %f\n' % (trainCnt+1, errorRate, trLoss))
        
        # 에러율이 가장 낮은 파라미터 저장
        if errorRate < bestErrorRate:
            bestParams = modelFull.get_weights()
            bestErrorRate = errorRate
            
        # 2000번의 반복 전에 8% 이하의 에러율이 나오면 즉시 종료
        if errorRate < 8.0:
            break;

        #학습 횟수 증가
        trainCnt += 1
        
        
    # 모델 파라미터를 에러율이 가장 낮았던 파라미터로 설정
    modelFull.set_weights(bestParams)
    
    # 로그파일 close
    fdTrainLog.close()
    print "\n'train_log.txt' file is created."
        
    # 모델 파라미터를 파일로 저장
    km.save_model(modelFull, 'best_param.h5')
    km.save_model(modelRelu1, 'best_param_for_feature_map.h5')
    print "'best_param.h5' file is saved."
    print "'best_param_for_feature_map.h5' file is saved."
    
    