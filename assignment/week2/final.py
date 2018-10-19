#-*- coding: utf-8 -*-
import Tkinter

map = []
map.append(range(1, 6))
map.append(range(6, 11))
map.append(range(11, 16))
map.append(range(16, 21))
map.append(range(21, 26))

# 막힌 곳은 0으로 표시
map[0][3] = 0
map[1][1] = 0
map[2][1] = 0
map[1][3] = 0
map[2][3] = 0
map[4][3] = 0



# 노드의 좌표로부터 이름 반환
def getNodeName(location):
    return map[location[0]][location[1]]
    
    
    
# 해당 노드로 이동 가능한지 확인
def isExist(location, toVisit, alreadyVisited):
    if location[0] < 0:
        return False
    
    if location[1] < 0:
        return False
    
    if location[0] > 4:
        return False

    if location[1] > 4:
        return False
        
        
    # 막힌 곳 판정
    if getNodeName(location) == 0:
        return False
        
    # 방문해야 할 목록에 이미 들어있는지 판정
    for i in range(len(toVisit)):
        if location[0:2] == toVisit[i][0:2]:
            return False
            
    # 이미 방문했던 곳 판정
    for i in range(len(alreadyVisited)):
        if location[0:2] == alreadyVisited[i][0:2]:
            return False
        
    return True
    
    
    
# 윈도우 콜백 클래스
class App:
    def __init__(self, master):
        # 맵을 그릴 캔버스 생성
        self.canvas = Tkinter.Canvas(master, width = 800, height = 600)
        self.canvas.pack()
        
        # 버튼 생성
        self.button = Tkinter.Button(master, text = 'Run', command = self.run)
        self.button.pack()
        
        # 맵 그리기
        for row in range(len(map)):
            for col in range(len(map[0])):
                if map[row][col] == 0:
                    fillColor = 'black'
                else:
                    fillColor = 'white'
                    
                self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = fillColor, outline = 'blue')
                self.canvas.create_text(col * 100 + 50, row * 100 + 50, text = map[row][col])
        
        
        # A* 초기화 ( row, col, g(n), h(n), f(n) )
        self.start = [2, 0, 0, 4, 4]
        
        self.route = []
        self.alreadyVisited = []
        self.toVisit = []

        self.toVisit.append(self.start)
    
    
    # sort함수에서 마지막요소(f(n))를 기준으로 소팅하기 위한 함수
    def myCmp(self, element1, element2):
        if element1[-1] > element2[-1]:
            return 1
        return -1
    
    
    # 목표점(goal)설정, H(n)을 구하기 위한 함수
    def calcHn(self, row, col):
        self.goal = [2, 4, 4, 0, 4]
        result = abs(row - self.goal[0]) + abs(col - self.goal[1])
        return result
        
    # 중간과정(탐색결과)들을 콘솔에 출력해주는 함수
    def printing(self, current, toVisit, alreadyVisited):
        print "Current node: ", getNodeName(current)
        
        visitable = []
        for node in toVisit:
            visitable.append("Node:" + str(getNodeName(node)) + " <f(n)=" + str(node[4]) + ", g(n)=" + str(node[2]) + ", h(n)=" + str(node[3]) + ">")
        print "Visitable Node: ", str(visitable)
        
        visited = []
        for node in alreadyVisited:
            visited.append(getNodeName(node))
        print "Already Visited Node: ", visited, "\n"



    # 목적지에 도착했을때 하나의 최종경로를 찾는 함수
    def searchFinalRoute(self, alreadyVisited):
        # 마지막 노드가 백트리킹 출발점
        finalRoute = []
        finalRoute.append(alreadyVisited.pop())

        # alreadyVisited 리스트의 마지막 노드부터 백트래킹 진행        
        while len(alreadyVisited) > 0:
            
            #이전 노드와 g(n) 값이 1씩 작아지는 노드를 찾아서 최종경로에 삽입
            node = alreadyVisited.pop()
            if finalRoute[-1][2] - 1 == node[2]:
                finalRoute.append(node)

                #출발점까지 왔다면 중단
                if getNodeName(finalRoute[-1]) == getNodeName(self.start):
                    break

        # 구한 노드들의 역순이 최종경로
        finalRoute.reverse()
        return finalRoute
        
           
        
    # run버튼을 누를때 마다 A* 알고리즘 반복 수행
    def run(self):
    
        # 더이상 방문할 노드가 없는데, 목적지가 아니면 실패, 맞으면 성공
        if len(self.toVisit) == 0:
            if self.route[-1] != getNodeName(self.goal):
                self.canvas.create_text(250, 550, text = "A* search Fail")
                print "Fail! The route can't be find."
                self.button.configure(text = "Finished", command="")                
                
            else:
                self.canvas.create_text(250, 550, text = "A* search Final Route: " + str(self.route))
                self.canvas.create_text(250, 570, text ="Searched Times: " + str(len(self.route)))
                print "Success! The result is printed in GUI."
                self.button.configure(text = "Finished", command="")
            
            
        # 시작하지 않았거나(찾은 루트가 없거나), 아직 목적지에 도착하지 못했을 경우
        elif len(self.route) == 0 or self.route[-1] != getNodeName(self.goal):
                        
            #toVisit에 있는 원소들을 각 리스트의 마지막 원소(f(n)값)을 기준으로 소팅
            self.toVisit.sort(self.myCmp) 
                    
            # f(n)이 가장 작은것(소팅 후 첫번째 원소)이 current
            current = self.toVisit.pop(0)

            # 현재 노드 빨간색으로 칠하기
            row = current[0]
            col = current[1]
            self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = 'red', outline = 'blue')
        
            # 현재 노드 이름 얻고, 이동한 루트에 추가, 이미 방문한 노드에 추가
            nodeName = getNodeName(current)
            self.route.append(nodeName)
            self.alreadyVisited.append(current)
           
            
            # 현재 노드의 자식 노드(인접 노드)를 방문해야 할 예비목록에 추가
            childList = []
            childList.append([current[0], current[1] - 1, current[2]+1, self.calcHn(current[0], current[1] - 1)])
            childList.append([current[0] + 1, current[1], current[2]+1, self.calcHn(current[0] + 1, current[1])])
            childList.append([current[0], current[1] + 1, current[2]+1, self.calcHn(current[0], current[1] + 1)])
            childList.append([current[0] - 1, current[1], current[2]+1, self.calcHn(current[0] - 1, current[1])])
            
            # 예비목록 중 갈 수 있는 노드인 경우에만 f(n)을 계산해서 toVisit 리스트에 추가
            for child in childList:
                if isExist(child, self.toVisit, self.alreadyVisited) == True:
                    child.append(child[2] + child[3])
                    self.toVisit.append(child)

            #중간 탐색결과 출력
            self.printing(current, self.toVisit, self.alreadyVisited)
            
            #목적지에 도착했는지 검사해서 도착했으면 결과출력
            if self.route[-1] == getNodeName(self.goal):
                finalRoute = self.searchFinalRoute(self.alreadyVisited)
                finalRouteName = []
                for node in finalRoute:
                    row = node[0]
                    col = node[1]
                    self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = 'green', outline = 'blue')
                    finalRouteName.append(getNodeName(node))
                self.canvas.create_text(250, 550, text = "A* search Final Route: " + str(finalRouteName))
                self.canvas.create_text(250, 570, text ="Searched Times: " + str(len(self.route)))
                print "Success! The result is printed in GUI."
                self.button.configure(text = "Finished", command="")
        
        
        
# 메인
root = Tkinter.Tk()
app = App(root)
root.mainloop()
