#-*- coding: utf-8 -*-
import sys

#프로그램 설명 출력
print "This program give a classified result of fish.\nIn input file, each line should be \"[body length](Tab)[tail length]\"\n"

try: #입력 파일 오픈
	fd = open('input_data.txt', 'r')
except IOError, fd: #파일 열기 실패시 예외처리해서 종료
    print "file open failed!\ninput_data.txt file is required."
    sys.exit(1)
except: #기타 에러는 unknown 에러메세지 출력 후 종료
    print "file open failed!\nunknown error is occured."
    sys.exit(1)

#입력파일 내용을 읽어들임
lines = fd.readlines()
fd.close()

#데이터의 갯수만큼 반복문 실행
for i in range(len(lines)): 
    lines[i] = lines[i].strip('\r\n').split('\t') #읽어들인 lines 리스트를 정리(각 리스트속에 리스트를 생성)

    #입력값이 형식(tab으로 구분된 2개 값)에 맞지 않으면 해당 라인은 에러로 처리   
    if len(lines[i]) != 2:
        lines[i].insert(0, 'Input line error')

    #몸길이가 85이하고 꼬리가 10이상이면 salmon으로 분류
    elif int(lines[i][0]) <= 85 and int(lines[i][1]) >= 10: 
        lines[i].append('salmon') #리스트 마지막에 어종결과 추가
    else:
        lines[i].append('seabass') #위 조건에서 salmon이 아닌 나머지 어종은 모두 seabass

try : #출력파일 열기(생성)
    fd = open('output_result.txt', 'w') 
except IOError, fd:  #파일 쓰기 실패시 예외처리해서 종료
    print "file write failed!\noutput_result.txt file can't be created."
    sys.exit(1)
except: #기타 에러는 unknown 에러메세지 출력 후 종료
    print "file write failed!\nunknown error is occured."
    sys.exit(1)

#에러 발생 여부를 기록하기 위한 변수    
errorOccured = False

#for문을 반복하며 lines 배열의 값들을 주어진 과제 형식에 맞게 기록
for i in range(len(lines)): 

    #위에서 에러로 분류된 라인은 에러로 기록하고 건너뜀
    if lines[i][0] == 'Input line error':
        fd.write(lines[i][0] + '\n')
        errorOccured = True
        continue

    #정상적인 입력은 결과 출력
    fd.write('body: ' + lines[i][0] + ' tail: ' + lines[i][1] + ' ==> ' + lines[i][2] + '\n')

fd.close()


#결과 출력
print "output_result.txt is created!\n"
if errorOccured == True:
    print "Some of lines has a error!\nEach line should be \"[integer](Tab)[integer]\""