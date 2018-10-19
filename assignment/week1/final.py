#-*- coding: utf-8 -*-
import sys

#���α׷� ���� ���
print "This program give a classified result of fish.\nIn input file, each line should be \"[body length](Tab)[tail length]\"\n"

try: #�Է� ���� ����
	fd = open('input_data.txt', 'r')
except IOError, fd: #���� ���� ���н� ����ó���ؼ� ����
    print "file open failed!\ninput_data.txt file is required."
    sys.exit(1)
except: #��Ÿ ������ unknown �����޼��� ��� �� ����
    print "file open failed!\nunknown error is occured."
    sys.exit(1)

#�Է����� ������ �о����
lines = fd.readlines()
fd.close()

#�������� ������ŭ �ݺ��� ����
for i in range(len(lines)): 
    lines[i] = lines[i].strip('\r\n').split('\t') #�о���� lines ����Ʈ�� ����(�� ����Ʈ�ӿ� ����Ʈ�� ����)

    #�Է°��� ����(tab���� ���е� 2�� ��)�� ���� ������ �ش� ������ ������ ó��   
    if len(lines[i]) != 2:
        lines[i].insert(0, 'Input line error')

    #�����̰� 85���ϰ� ������ 10�̻��̸� salmon���� �з�
    elif int(lines[i][0]) <= 85 and int(lines[i][1]) >= 10: 
        lines[i].append('salmon') #����Ʈ �������� ������� �߰�
    else:
        lines[i].append('seabass') #�� ���ǿ��� salmon�� �ƴ� ������ ������ ��� seabass

try : #������� ����(����)
    fd = open('output_result.txt', 'w') 
except IOError, fd:  #���� ���� ���н� ����ó���ؼ� ����
    print "file write failed!\noutput_result.txt file can't be created."
    sys.exit(1)
except: #��Ÿ ������ unknown �����޼��� ��� �� ����
    print "file write failed!\nunknown error is occured."
    sys.exit(1)

#���� �߻� ���θ� ����ϱ� ���� ����    
errorOccured = False

#for���� �ݺ��ϸ� lines �迭�� ������ �־��� ���� ���Ŀ� �°� ���
for i in range(len(lines)): 

    #������ ������ �з��� ������ ������ ����ϰ� �ǳʶ�
    if lines[i][0] == 'Input line error':
        fd.write(lines[i][0] + '\n')
        errorOccured = True
        continue

    #�������� �Է��� ��� ���
    fd.write('body: ' + lines[i][0] + ' tail: ' + lines[i][1] + ' ==> ' + lines[i][2] + '\n')

fd.close()


#��� ���
print "output_result.txt is created!\n"
if errorOccured == True:
    print "Some of lines has a error!\nEach line should be \"[integer](Tab)[integer]\""