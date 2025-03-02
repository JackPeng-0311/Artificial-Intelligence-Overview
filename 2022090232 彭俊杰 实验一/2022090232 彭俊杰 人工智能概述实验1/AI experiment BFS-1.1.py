"""
author:Jack Peng
time:2024-04-09 22:33:09
version:V-1.1.2
file name:AI experiment BFS-1.1
"""
import operator
import copy
import time
import numpy as np

# 状态对象
class Object:
    def __init__(self):
        self.Matrix = []
        self.Fn = 0  # 估值函数
        self.Dn = 0  # 深度
        self.camefrom = 0  # 记录先前操作
        self.Father = Object  # 用于回溯


# 操作函数
def Operation(i, j, camefrom):  # 矩阵的0的坐标、先前操作
    operationNum = []  # 操作矩阵
    # 防止变回原先状态
    if i == 0 or i == 1:
        if camefrom != 1:
            operationNum.append(2)
    if i == 1 or i == 2:
        if camefrom != 2:
            operationNum.append(1)
    if j == 0 or j == 1:
        if camefrom != 3:
            operationNum.append(4)
    if j == 1 or j == 2:
        if camefrom != 4:
            operationNum.append(3)
    return operationNum


def up():
    return (-1, 0)


def down():
    return (1, 0)


def left():
    return (0, -1)


def right():
    return (0, 1)


# 矩阵元素寻找函数
def find(currentMatrix, x):
    for i in range(3):
        for j in range(3):
            if currentMatrix[i][j] == x:
                return [i, j]  # 返回元素x坐标


# 变换矩阵函数
def move(currentMatrix, operationNum):
    newMatrix = copy.deepcopy(currentMatrix)  # 保留原矩阵
    i0, j0 = find(currentMatrix, 0)  # 或者写成X0[i，j]
    i, j = 0, 0
    # 交换两个坐标
    if (operationNum == 1):  # up
        i, j = up()
    if (operationNum == 2):  # down
        i, j = down()
    if (operationNum == 3):  # left
        i, j = left()
    if (operationNum == 4):  # right
        i, j = right()
    tmp = currentMatrix[i0 + i][j0 + j]  # 即将交换的非0元素
    newMatrix[i0][j0] = tmp
    newMatrix[i0 + i][j0 + j] = 0
    return newMatrix  # 子状态矩阵


# wn计算函数
def WrongNum(currentMatirx, endMatrix):
    wn = 0
    currentMatirx = np.array(currentMatirx)
    for i in range(3):
        for j in range(3):
            if (currentMatirx[i][j] != endMatrix[i][j]) and currentMatirx[i][j] != 0:
                wn += 1
    return wn


# 曼哈顿距离计算函数
def LengthNum(currentMatrix, endMatrix):
    ln = 0
    for i in range(3):
        for j in range(3):
            endX = find(endMatrix, currentMatrix[i][j])
            ln += abs(endX[0] - i) + abs(endX[1] - j)
    return ln


# 打印函数
def show(currentMatrix):
    for i in range(3):
        print(currentMatrix[i])
    print("-" * 10)


# 逆序数计算函数
def reverseNum(Matrix):
    num = 0
    Matrix = sum(Matrix, [])  # 转换成一维数组
    for i in range(len(Matrix)):
        if Matrix[i] != 0:
            for j in range(i):
                if Matrix[j] > Matrix[i]:
                    num += 1
    return num


# 判断有解函数
def judge(startMatrix, endMatrix):
    rNum1 = reverseNum(startMatrix)
    rNum2 = reverseNum(endMatrix)
    if rNum1 % 2 == rNum2 % 2:  # 同奇偶
        return True
    else:
        return False


# 主程序
start = time.time()
openQuene = []
closeList = []
endMatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # 自己输入
countDn = 0

startObject = Object()
startObject.Matrix = [[0, 2, 3], [6, 7, 4], [1, 8, 5]]  # 自己定义起始状态
startObject.Fn = countDn + WrongNum(startObject.Matrix, endMatrix) + LengthNum(startObject.Matrix, endMatrix)  # 初始估值函数
openQuene.append(startObject)  # 初始对象加入open表

print("起始状态逆序数为：", end='')
print(reverseNum(startObject.Matrix))
print("目标状态逆序数为：", end='')
print(reverseNum(endMatrix))

# BFS算法
if judge(startObject.Matrix, endMatrix):  # 判断是否有解
    while 1:
        if operator.eq(openQuene[0].Matrix, endMatrix):  # 判断open栈顶是否为目标
            closeList.append(openQuene[0])  # 压如close表
            endList = [endMatrix]  # 回溯列表
            father = closeList[-1].Father
            while father.Dn >= 1:
                endList.append(father.Matrix)
                father = father.Father
            print("变换成功，一共需要" + str(openQuene[0].Dn) + "次变换")
            print("初始状态：")
            show(startObject.Matrix)
            flag = 1
            for i in reversed(endList):
                print("第" + str(flag) + "次变换：")
                show(i)
                flag += 1
            break
        else:
            i0, j0 = find(openQuene[0].Matrix, 0)
            operation = Operation(i0, j0, openQuene[0].camefrom)
            nextObjectList = []  # 暂存列表
            for i in operation:
                # 对子对象因素赋值
                nextMatrix = move(openQuene[0].Matrix, i)
                nextObject = Object()
                nextObject.Matrix = nextMatrix
                nextObject.Dn = openQuene[0].Dn + 1
                nextObject.camefrom = i
                nextObject.Father = openQuene[0]
                nextObjectList.append(nextObject)
            # 将open表尾压如close表
            closeList.append(openQuene[0])
            del openQuene[0]
            for i in nextObjectList:
                openQuene.append(i)  # 在表尾插入元素

else:
    print("无解")
end = time.time()
print("运行时间：%s"%(end-start))