# -*- coding: utf-8 -*-
import struct as st
import numpy as np
import numpy.random as nr
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku
import matplotlib.pyplot as plt

# MNIST 데이터 경로
_SRC_PATH = u'..\\'
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL

# 기타 필요한 상수 정의
_N_CLASS = 10



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
    # 테스트 데이터 / 레이블 로드
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    return tsDataList, tsLabelList

    
# indexList 에서, 원하는 Class에 옳게 분류된 샘플의 인덱스를 amout 개 만큼 찾아서 리스트로 반환함
def getIndexNum(indexList, numClass, classificationResult, answerLabel, amount):
    selectedIndex = []
    
    for i in indexList:
        if (answerLabel[i] == numClass) & (answerLabel[i] == classificationResult[i]):
            selectedIndex.append(i) # 조건을 만족하는 인덱스 번호 저장
            
            if len(selectedIndex) == amount: # amount 개수 만큼 저장했으면 반복문 탈출
                break;
                
    return selectedIndex
    
    
def drawFeatMap(dstFn, featMap):

    fig, ax = plt.subplots()
    ax.imshow(featMap, cmap='gray')
    plt.savefig(dstFn)
    
    # flush
    plt.cla()
    plt.clf()
    plt.close()
    
    
# 프로그램 메인함수    
if __name__ == '__main__':

    nr.seed(12345)  # random seed 설정
    tsDataList, tsLabelList = loadMNIST() # MNIST 데이터셋 로드
    print "tsDataList: ", tsDataList.shape
    print "tsLabelList: ", tsLabelList.shape
    print "MNIST Data load done.\n"
    
    # 모델 파라미터 로드
    modelFull = km.load_model('best_param.h5')
    modelRelu1 = km.load_model('best_param_for_feature_map.h5')    
    
    # 테스트 준비
    fdTestLog = open('test_output.txt', 'w') 
    
    # 테스트 진행
    res = modelFull.predict(tsDataList, batch_size=100)
    res2 = modelRelu1.predict(tsDataList, batch_size=100)
    print 'Predict done.\n'
    print "-----Test Start-----\n"
    
    classificationResult = np.argmax(res, axis=1)
    answerLabel = np.argmax(tsLabelList, axis=1)
    
    # 각 샘플별 분류결과 로그파일에 저장, 500개에 1개씩은 콘솔에도 분류 결과 출력
    for i in range(tsDataList.shape[0]):
        if classificationResult[i]== answerLabel[i]:
            resultMsg = 'Correct'
        else:
            resultMsg = 'Wrong  '
        fdTestLog.write('%5d-th sample ---> %s (Classification Result: %d / Answer: %d)\n' % (i+1, resultMsg, classificationResult[i], answerLabel[i]))
            
        if i % 500 == 0:
            print '%5d-th sample ---> %s (Classification Result: %d / Answer: %d)' % (i+1, resultMsg, classificationResult[i], answerLabel[i])
    
    
    # 모든 샘플에 대한 최종 오류율 계산 및 출력
    testResult = (classificationResult == answerLabel).astype('float32')
    testResultErrorRate = ((testResult.shape[0]-np.sum(testResult)) / testResult.shape[0]) * 100
    
    print '\nTotal Error Rate: %.2f%% (%5d / %5d)' % (testResultErrorRate, testResult.shape[0]-np.sum(testResult), testResult.shape[0])
    fdTestLog.write('\nTotal Error Rate: %.2f%% (%5d / %5d)\n' % (testResultErrorRate, testResult.shape[0]-np.sum(testResult), testResult.shape[0]))
    
    fdTestLog.close()
    print "\n'train_log.txt' file is created.\n\nDrawing Feature Map..."
    
    
    # Correct 하게 분류된 샘플 중에서, 특정 클래스의 인덱스를 랜덤하게 얻음
    indexList = np.arange(tsDataList.shape[0]) # 10000개 인덱스 numpy List 생성
    nr.shuffle(indexList)
    class5Index = getIndexNum(indexList, 5, classificationResult, answerLabel, 3) # 숫자 '5'로 옳게 분류된 샘플의 인덱스 3개를 얻음
    class7Index = getIndexNum(indexList, 7, classificationResult, answerLabel, 3) # 숫자 '7'로 옳게 분류된 샘플의 인덱스 3개를 얻음
    
    
    # 숫자 '5' 샘플 3개 출력
    for sampleIndex in class5Index:
        for filterIndex in range(res2.shape[3]):
            dstFn = '5_%d_%d.png' % (sampleIndex, filterIndex)
            featMap = res2[sampleIndex, :, :, filterIndex]
            drawFeatMap(dstFn, featMap)
            
    # 숫자 '7' 샘플 3개 출력
    for sampleIndex in class7Index:
        for filterIndex in range(res2.shape[3]):
            dstFn = '7_%d_%d.png' % (sampleIndex, filterIndex)
            featMap = res2[sampleIndex, :, :, filterIndex]
            drawFeatMap(dstFn, featMap)
            
    print "Feature Map Images are saved."