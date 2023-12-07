from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game, capture
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState: capture.GameState):
        self.start = gameState.getAgentPosition(self.index)

        # some info we want to track
        self.layout = gameState.data.layout

        # self.foodToEat = self.getFood(gameState)
        # self.foodToDefend = self.getFoodYouAreDefending(gameState)
        #
        # self.capsulesToEat = self.getCapsules(gameState)

        self.red = gameState.isOnRedTeam(self.index)

        self.walls = gameState.getWalls()
        self.myAgentsIndices = self.getTeam(gameState)

        self.enemyAgentsIndices = self.getOpponents(gameState)

        self.allCoordinates = [(a, b) for a in range(self.layout.width)
                               for b in range(self.layout.height)]
        self.allCoordsNoWalls = [(a, b) for (a, b) in self.allCoordinates if not self.walls[a][b]]

        if not self.red:  # if blue, the right half is ours
            self.homeTerritoryNoWalls = [(a, b) for (a, b) in self.allCoordsNoWalls if a >= self.layout.width // 2]
            self.oppTerritoryNoWalls = [(a, b) for (a, b) in self.allCoordsNoWalls if a < self.layout.width // 2]
        else:
            self.homeTerritoryNoWalls = [(a, b) for (a, b) in self.allCoordsNoWalls if a < self.layout.width // 2]
            self.oppTerritoryNoWalls = [(a, b) for (a, b) in self.allCoordsNoWalls if a >= self.layout.width // 2]

        self.foodToEat = len(self.getFood(gameState).asList())
        self.lastDepositedFood = 0
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState: capture.GameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        self.foodToEat = len(self.getFood(gameState).asList())
        foodLeft = self.foodToEat

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState: capture.GameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState: capture.GameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState: capture.GameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState: capture.GameState, action):

        offenseFeatures = util.Counter()
        defenseFeatures = util.Counter()

        successor = self.getSuccessor(gameState, action)
        newPosition = successor.getAgentPosition(self.index)
        newState = successor.getAgentState(self.index)


        self.homeTerritoryNoWalls.sort(key=lambda pos: self.getMazeDistance(newPosition, pos))
        foodList = self.getFood(successor).asList()
        capList = self.getCapsules(successor)


        capsulesToEat = self.getCapsules(successor)

        offenseFeatures['successorScore'] = -len(foodList)  # self.getScore(successor)
        foodInStomach = self.foodToEat - len(foodList) - self.lastDepositedFood
        offenseFeatures['capsulesScore'] = -len(capList)
        enemyPacmanIndices = [i for i in self.enemyAgentsIndices if successor.getAgentState(i).isPacman]
        enemyGhostIndices = [i for i in self.enemyAgentsIndices if not successor.getAgentState(i).isPacman]

        h = min([self.getMazeDistance(newPosition, homePos) for homePos in self.homeTerritoryNoWalls])
        d = min([self.getMazeDistance(newPosition, successor.getAgentPosition(i)) for i in enemyGhostIndices])

        if newState.isPacman and foodInStomach >= 2:
            print(foodInStomach)
            print(self.foodToEat, self.lastDepositedFood, foodInStomach)
            f = util.Counter()
            f['distToHome'] = h
            f['distToClosestGhost'] = 1/d
            return f


        prevObs = self.getPreviousObservation()
        if prevObs is not None:
            if newPosition == prevObs.getAgentPosition(self.index):
                offenseFeatures['notMoving'] = 1
        else:
            offenseFeatures['notMoving'] = 0

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minFoodDistance = min([self.getMazeDistance(newPosition, food) for food in foodList])
            minCapDistance = min([self.getMazeDistance(newPosition, cap) for cap in capList]) if len(capList) > 0 else 0
            offenseFeatures['distanceToFood'] = minFoodDistance
            offenseFeatures['distanceToCapsule'] = minCapDistance

        if len(enemyGhostIndices) == 0:
            offenseFeatures['distToClosestGhost'] = 0
        else:
            offenseFeatures['distToClosestGhost'] = 1 / (d + 0.000001)

        offenseFeatures['successorScore'] = -len(foodList)
        offenseFeatures['distanceToFood'] = minFoodDistance
        offenseFeatures['isCorner'] = 0

        x, y = newPosition
        wallsSet = set(tuple(wall) for wall in self.walls)
        if {(x + 1, y), (x, y + 1), (x, y - 1)}.issubset(wallsSet) or {(x, y + 1), (x + 1, y), (x - 1, y)}.issubset(
                wallsSet) \
                or {(x - 1, y), (x, y + 1), (x, y - 1)}.issubset(wallsSet) or {(x, y - 1), (x + 1, y),
                                                                               (x - 1, y)}.issubset(wallsSet):
            offenseFeatures['isCorner'] = 1

        # foodInStomach management here
        if not newState.isPacman and foodInStomach != 0:
            if self.observationHistory[-1] is not None:
                lastPos = gameState.getAgentPosition(self.index)
                if self.getMazeDistance(lastPos, newPosition) > 2:  # I died as a pacman having eaten at least 1 food
                    self.lastDepositedFood = 0
                else:  # I deposited food
                    self.lastDepositedFood = foodInStomach
        # allFeatures = {**offenseFeatures, **defenseFeatures}

        # if not newState.isPacman:
        #     offenseFeatures['distToClosestGhost'] = d
        return offenseFeatures

    def getWeights(self, gameState: capture.GameState, action):

        defenseWeights = util.Counter()
        offenseWeights = util.Counter()
        offenseWeights['successorScore'] = 70
        offenseWeights['capsulesScore'] = 0
        offenseWeights['distToClosestGhost'] = -50
        offenseWeights['notMoving'] = - 1
        offenseWeights['distToHome'] = -1000
        offenseWeights['distanceToCapsule'] = -5

        offenseWeights['distanceToFood'] = - 1
        offenseWeights['isCorner'] = 0

        isPacman = self.getSuccessor(gameState, action).getAgentState(self.index).isPacman

        return offenseWeights  # if isPacman else defenseWeights


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState: capture.GameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState: capture.GameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
