from Tkinter import *

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
        
    # 이미 방문해야 할 목록에 들어있는지 판정
    if location in toVisit:
        return False
        
    # 이미 방문했던 곳 판정
    if location in alreadyVisited:
        return False
        
    return True
    
    
    
# 윈도우 콜백 클래스
class App:
    def __init__(self, master):
        # 맵을 그릴 캔버스 생성
        self.canvas = Canvas(master, width = 800, height = 600)
        self.canvas.pack()
        
        # 버튼 생성
        button = Button(master, text = 'run', command = self.run)
        button.pack()
        
        # 맵 그리기
        for row in range(len(map)):
            for col in range(len(map[0])):
                if map[row][col] == 0:
                    fillColor = 'black'
                else:
                    fillColor = 'white'
                    
                self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = fillColor, outline = 'blue')
                self.canvas.create_text(col * 100 + 50, row * 100 + 50, text = map[row][col])
        
        # A* 초기화
        start = [2, 0]
        node = getNodeName(start)
        
        self.alreadyVisited = []
        self.toVisit = []

        self.toVisit.append(start)
        
        
        
    # A* 알고리즘 반복 수행
    def run(self):
        # 앞으로 방문해야 할 노드가 남아있으면 루프 반복
        if len(self.toVisit) != 0:
            # 방문해야 할 노드 목록의 첫 번째로 현재 노드 이동
            current = self.toVisit.pop(-1)
            
            # 현재 노드 칠하기
            row = current[0]
            col = current[1]
            self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = 'red', outline = 'blue')
            
            nodeName = getNodeName(current)
            
            # 현재 노드의 자식 노드(인접 노드)를 방문해야 할 목록에 추가
            childList = []
            childList.append([current[0], current[1] - 1])
            childList.append([current[0] + 1, current[1]])
            childList.append([current[0], current[1] + 1])
            childList.append([current[0] - 1, current[1]])
            
            for child in childList:
                # 갈 수 있는 노드인 경우에만 추가
                if isExist(child, self.toVisit, self.alreadyVisited) == True:
                    self.toVisit.append(child)
                
            # 이미 방문한 노드에 현재 노드 추가
            self.alreadyVisited.append(current)
            
            print nodeName, self.toVisit, self.alreadyVisited

        
        
# 메인
root = Tk()
app = App(root)
root.mainloop()
