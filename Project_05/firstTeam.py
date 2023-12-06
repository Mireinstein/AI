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

        self.walls = gameState.getWalls()
        self.myAgentsIndices = self.getTeam(gameState)

        self.enemyAgentsIndices = self.getOpponents(gameState)

        self.allCoordinates = [(a, b) for a in range(self.layout.width)
                               for b in range(self.layout.height)]
        self.allCoordsNoWalls = [(a, b) for (a, b) in self.allCoordinates if not self.walls[a][b]]

        if not self.red:  # if blue, the right half is ours
            self.homeTerritoryNoWalls = [(a, b) for (a, b) in self.allCoordsNoWalls if a >= self.layout.width]
            self.oppTerritoryNoWalls = [(a, b) for (a, b) in self.allCoordsNoWalls if a < self.layout.width]
        else:
            self.homeTerritoryNoWalls = [(a, b) for (a, b) in self.allCoordsNoWalls if a < self.layout.width]
            self.oppTerritoryNoWalls = [(a, b) for (a, b) in self.allCoordsNoWalls if a >= self.layout.width]

        self.foodInStomach = 0
        self.myEnemyToHomeCrossings = 0

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
        features = util.Counter()
        currentPosition = gameState.getAgentPosition(self.index)

        self.homeTerritoryNoWalls.sort(key=lambda pos: self.getMazeDistance(currentPosition, pos))
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()

        capsulesToEat = self.getCapsules(gameState)

        if (self.getCurrentObservation().getAgentPosition(self.index) in self.homeTerritoryNoWalls)\
                and (self.getPreviousObservation().getAgentPosition(self.index) in self.oppTerritoryNoWalls):
            



        features['successorScore'] = -len(foodList)  # self.getScore(successor)
        features['capsulesScore'] = -len(capsulesToEat)
        features['goHome'] = 0

        if features['successorScore'] >= 5: # not exactly what I should write
            features['goHome'] = self.getMazeDistance(currentPosition, self.homeTerritoryNoWalls[0])

        print(features['successorScore'])

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState: capture.GameState, action):

        weights = util.Counter()
        weights['successorScore'] = 100
        weights['capsulesScore'] = 500
        weights['goHome'] = 0

        if self.getFeatures(gameState)['successorScore'] >= 3:
            weights['successorScore'] = 1
            weights['goHome'] = 100

        weights['distanceToFood'] = -1


        return weights


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
