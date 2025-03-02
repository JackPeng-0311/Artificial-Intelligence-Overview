from util import manhattanDistance
from game import Directions
import random, util
from game import Agent


class ReflexAgent(Agent):

    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action): # 原本是用的.getScore(),无法避免墙壁的影响
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood() # 效果不好/我后继的位置可能和豆子覆盖，此时是最佳的，但是用newfood不会出现后继覆盖豆子
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostPos = successorGameState.getGhostPositions()  # 幽灵坐标

        # 权重
        GHOST_WEIGHT = 1500
        FOOD_WEIGHT = 700

        # 食物因素
        nowFood = currentGameState.getFood()
        food_D = []
        for foodPos in nowFood.asList():
            food_D.append(manhattanDistance(newPos, foodPos))
        fScore = FOOD_WEIGHT / (min(food_D) + 1)

        # 幽灵因素
        ghost_D = min(manhattanDistance(newPos, i) for i in ghostPos)
        gScore = - GHOST_WEIGHT / (ghost_D + 1)  # 使用距离倒数来放大接近幽灵时的负面影响

        total_score = gScore + fScore

        # 强制启动/在openClassic没有什么作用，基本不会停的
        #if action == "Stop" and ghost_D > 2:
            #total_score -= 1e100

        return total_score


def scoreEvaluationFunction(currentGameState): # EV默认指向
    return currentGameState.getScore() # 即getScore

class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn = 'better', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        def maxV(gameState, depth):
            v = -1e10
            if depth == self.depth or gameState.isLose() or gameState.isWin(): # 判断是否到达terminal层
                return self.evaluationFunction(gameState)
            for i in gameState.getLegalActions(0): # PacmanLA
                v = max(v, minV(gameState.generateSuccessor(0, i), depth, 1))
            return v

        def minV(gameState, depth, agentIndex):
            v = 1e10
            if gameState.isWin() or gameState.isLose(): # 这里不能加depth条件
                return self.evaluationFunction(gameState)
            for i in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v = min(v, maxV(gameState.generateSuccessor(agentIndex, i), depth + 1)) # 在每次递归调用max的时候深度+1
                else:
                    v = min(v, minV(gameState.generateSuccessor(agentIndex, i), depth, agentIndex + 1))
            return v

        def Value(gameState):
            LegalAction = gameState.getLegalActions(0)
            Max = -1e10
            action = 0
            if "Stop" in LegalAction:
                LegalAction.remove("Stop")
            for i in LegalAction:
                v = minV(gameState.generateSuccessor(0, i), 0, 1)
                if (v > Max): #记录对应动作
                    Max = v
                    action = i
            return action

        return Value(gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        def maxV(gameState, agentIndex, depth, alpha, beta): # 只更新alpha
            v = -1e10
            LegalAction = gameState.getLegalActions(0)
            if "Stop" in LegalAction:
                LegalAction.remove("Stop")
            for i in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, i)
                v= max(v, Value(successor, agentIndex, depth, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minV(gameState, agentIndex, depth, alpha, beta): # 只更新beta
            v = 1e10
            LegalAction = gameState.getLegalActions(0)
            if "Stop" in LegalAction:
                LegalAction.remove("Stop")
            for i in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, i)
                if agentIndex == gameState.getNumAgents() - 1:
                    v = min(v, Value(successor, 0, depth, alpha, beta))
                else:
                    v = min(v, Value(successor, agentIndex+1, depth, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        def Value(gameState, agentIndex, depth, alpha, beta):
            legalActions = gameState.getLegalActions(agentIndex)
            if len(legalActions) == 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                depth = depth + 1
                if depth == self.depth:
                    return self.evaluationFunction(gameState)
                else:
                    return maxV(gameState, agentIndex, depth, alpha, beta)
            elif agentIndex > 0:
                return minV(gameState, agentIndex, depth, alpha, beta)

        def getAction(gameState):
            alpha = -1e10
            beta = 1e10
            v = -1e10
            action = 0
            for i in gameState.getLegalActions(0):
                value = Value(gameState.generateSuccessor(0, i), 1, 0, alpha, beta)  #index=1,depth=0
                if value>v: #choose max v->action
                    v = value
                    action = i
                alpha=max(alpha,v) # 更新alpha
            return action

        return getAction(gameState)

def betterEvaluationFunction(currentGameState): #对当前状态评分，用于minimax评估呗
    nowPos = currentGameState.getPacmanPosition()
    nowFood = currentGameState.getFood()
    nowGhostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    ScaredTimes = [ghostState.scaredTimer for ghostState in nowGhostStates]

    # 权重
    FOOD_WEIGHT = 5  # 豆基础值
    GHOST_WEIGHT = 10  # 幽灵基值
    CAPSULE_WEIGHT = 100.0 # 胶囊基值

    #计算最近的豆的影响
    if len(nowFood.asList()) > 0:
        Food_D = (min([manhattanDistance(nowPos, food) for food in nowFood.asList()]))
        foodScore = FOOD_WEIGHT / Food_D
    else:
        foodScore = FOOD_WEIGHT

    # 评价幽灵的影响
    Ghost_D = min([manhattanDistance(nowPos, ghostState.configuration.pos) for ghostState in nowGhostStates])
    if Ghost_D > 5:
        ghostScore = GHOST_WEIGHT / Ghost_D
    elif Ghost_D != 0:
        if Ghost_D < 3 and ScaredTimes[0] > 5: # 追击幽灵
            ghostScore = 1e6 / Ghost_D
        else:
            ghostScore = - GHOST_WEIGHT / Ghost_D
    else:
        ghostScore = GHOST_WEIGHT

    # 胶囊影响
    '''capsule_D = []
    for i in capsules:
        capsule_distance = manhattanDistance(nowPos, i)
        capsule_D.append(capsule_distance)
    capsule_D.sort()
    if len(capsule_D) != 0:
        if capsule_D[0] <= 5 and Ghost_D >= 7:  # 假设我们只在5个单位内，且较为安全时考虑胶囊
            CAPSULE_WEIGHT = 1e4  # 改变胶囊权重
        capsuleScore = CAPSULE_WEIGHT / capsule_D[0]
    else:
        capsuleScore = CAPSULE_WEIGHT'''

    return currentGameState.getScore() + ghostScore + foodScore# + capsuleScore

# Abbreviation
better = betterEvaluationFunction


