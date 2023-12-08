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

    # We'll store as much info as we can in register initial stat;
    # and incrementally generate as many features as we can from this
    # With time, maybe we'll use all of these

    def registerInitialState(self, gameState: capture.GameState):
        CaptureAgent.registerInitialState(self, gameState)
        if self.red:
            self.registerTeam(gameState.getRedTeamIndices())
        else:
            self.registerTeam(gameState.getBlueTeamIndices())
        self.start = gameState.getAgentPosition(self.index)

        # data concerning grid
        self.layout = gameState.data.layout
        self.walls = gameState.getWalls()
        self.allCoordinates = [(a, b) for a in range(self.layout.width)
                               for b in range(self.layout.height)]
        self.allCoordsNoWalls = [(a, b) for (a, b) in self.allCoordinates if not self.walls[a][b]]

        # concerning food
        self.foodToDefendList = self.getFoodYouAreDefending(gameState).asList()
        self.foodToEatList = self.getFood(gameState).asList()
        self.totalFoodToEat = len(self.foodToEatList)
        self.totalFoodToDefend = len(self.foodToDefendList)

        # concerning capsules
        self.capsulesToEatList = self.getCapsules(gameState)
        self.totalCapsulesToEat = len(self.capsulesToEatList)
        self.capsulesToDefendList = self.getCapsulesYouAreDefending(gameState)
        self.totalCapsulesToDefend = len(self.capsulesToDefendList)

        # concerning team belonging and region
        for i in self.agentsOnTeam:
            if i != self.index:
                self.partnerIndex = i
        self.enemyAgentsIndices = self.getOpponents(gameState)
        halfWidth = gameState.data.layout.width // 2
        self.homeWidthBoundary = halfWidth if self.red else halfWidth + 1

        # bookkeeping and history-tracking
        self.actionHistory = [] # useless so far, ...
        self.foodThreshold = 0  # hardcoded, will experiment with this; might find a way to make it depend on total food available and layout

        # Q-learning parameters
        self.foodRewardMultiplier = 1  # might also make this dynamic
        self.discount = .9
        self.alpha = .001
        self.epsilon = 0

    def chooseActionComponent(self, gameState: capture.GameState):
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        self.foodToEat = len(self.getFood(gameState).asList())
        foodLeft = self.foodToEat

        # get a random action if food >= 2
        bestAction = random.choice(bestActions)

        # if this is the case, we have a new best action
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist

        # these capsules man:
        # anyway, here I am forcing the best action to always be eating a capsule if we can
        # it will mess up the q Valuation --might put this outside
        legalPositions = {self.getNextPosition(gameState.getAgentPosition(self.index), a): a for a in actions}
        for capsulePos in self.getCapsules(gameState):
            if capsulePos in legalPositions:
                bestAction = legalPositions[capsulePos]
        return bestAction

    def chooseAction(self, gameState: capture.GameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        bestAction = self.chooseActionComponent(gameState)
        nextState = gameState.generateSuccessor(self.index, bestAction)

        reward = self.getReward(gameState, bestAction, nextState)

        self.updateWeights(gameState, bestAction, nextState, reward)
        if util.flipCoin(self.epsilon):
            return random.choice(gameState.getLegalActions(self.index))
        else:
            return bestAction

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

    def getNextPosition(self, position, action):
        x, y = position
        if action == Directions.NORTH:
            return (x, y + 1)
        elif action == Directions.SOUTH:
            return (x, y - 1)
        elif action == Directions.EAST:
            return (x + 1, y)
        elif action == Directions.WEST:
            return (x - 1, y)
        else:
            return position

    def updateWeights(self, gameState: capture.GameState, action, nextState, reward):
        self.alpha *= .5
        oldQValue = self.evaluate(gameState, action)

        nextAction = self.chooseActionComponent(nextState)
        nextQValue = self.evaluate(nextState, nextAction)

        difference = reward + (self.discount * nextQValue) - oldQValue

        curFeatures = self.getFeatures(gameState, action)
        curWeights = self.getWeights(gameState)

        if curFeatures is not None and curWeights is not None:
            for feature in curFeatures:
                self.weights[feature] += (self.alpha * difference * curFeatures[feature])
        return

    def evaluate(self, gameState: capture.GameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState)
        return features * weights

    # temporary

    def getFeatures(self, gameState: capture.GameState, action):
        """
        Returns a counter of features for the state
        """
        pass

    def getWeights(self, gameState):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        pass

    def getReward(self, gameState, action, successor):
        """
        To determine the external reward
        :param successor:
        :param action:
        :param gameState:
        :return:
        """
        return 0


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    # Number all your features so we can keep track
    def registerInitialState(self, gameState: capture.GameState):
        self.weights = util.Counter({
            'successorScore': 100,
            'disToFood': -10,
            'disToGhost': 1000,
            'disToCap': -40,
            # 'capScore': 10,
            'returnHome': -5000,
            'isDeadEnd': -2,
        })
        ReflexCaptureAgent.registerInitialState(self, gameState)

    def getFeatures(self, gameState: capture.GameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()

        newPosition = successor.getAgentPosition(self.index)

        # Feature 1: Distance To The Nearest Food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minFoodDistance = min([self.getMazeDistance(newPosition, food) for food in foodList])
            features['disToFood'] = minFoodDistance

        # Feature 2: from baseline also
        features['successorScore'] = -len(foodList)

        # Feature 3: Distance To The Nearest Capsule
        capsulePositions = self.getCapsules(successor)
        if len(capsulePositions) > 0:  # I guess we'll double-check everything now
            minCapDistance = min([self.getMazeDistance(newPosition, capsule) for capsule in capsulePositions])
            features['disToCap'] = minCapDistance
        if not successor.getAgentState(self.index).isPacman:
            return features

        # Feature 4: Distance To The Nearest Ghost
        opps = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        # with the way I wrote this so far, this cannot return None since it runs when we are Pacman
        oppGhosts = [opp for opp in opps if not opp.isPacman and opp.getPosition() is not None]

        oppGhostPositions = [ghost.getPosition() for ghost in oppGhosts]
        distsToGhosts = [self.getMazeDistance(newPosition, ghostPos) for ghostPos in oppGhostPositions]

        closestGhostDist = -sys.maxsize
        if len(distsToGhosts) > 0:
            closestGhostDist = min(distsToGhosts)

        ghostScaredTimer = max([ghost.scaredTimer for ghost in oppGhosts])
        closestGhostDist = - (ghostScaredTimer * closestGhostDist // self.layout.width)
        currentThreshold = self.foodThreshold - ((ghostScaredTimer // self.layout.width)*self.foodThreshold)
        features['disToGhost'] = closestGhostDist

        # Feature 6: I'm still having trouble eating the capsules or I'm freezing up after I eat one.
        # features['capScore'] = -len(capsulePositions)

        # Feature 6: there was a variable all along lol
        foodInStomach = gameState.getAgentState(self.index).numCarrying
        if foodInStomach > currentThreshold:
            features['returnHome'] = self.getMazeDistance(newPosition, self.start)
            features['disToGhost'] *= 2

        # Feature 7: Avoid dead ends
        # legalActionsInSuccessor = successor.getLegalActions(self.index)
        # if len(legalActionsInSuccessor) == 2:
        #     features['isDeadEnd'] = 1
        return features

    def getReward(self, gameState: capture.GameState, action, nextState):
        return self.foodReward(gameState, action, nextState) + self.wasEatenReward(gameState)

    def foodReward(self, gameState: capture.GameState, action, nextState):
        current_position = gameState.getAgentPosition(self.index)
        # if we eat food
        if action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            next_position = self.getNextPosition(current_position, action)
            food = self.getFood(nextState)
            if food[next_position[0]][next_position[1]]:
                return 10
            if ([next_position[0]], [next_position[1]]) in self.getCapsules(nextState):
                return 20


        if not self.getFood(gameState)[current_position[0]][current_position[1]] and \
                not self.getFoodYouAreDefending(gameState)[current_position[0]][current_position[1]]:
            return -50

        successor = self.getSuccessor(gameState, action)
        current_distance = self.getMazeDistance(self.start, current_position)
        successor_distance = self.getMazeDistance(self.start, successor.getAgentPosition(self.index))
        if successor_distance <= current_distance:
            return -1000
        return 0

    def wasEatenReward(self, gameState: capture.GameState):
        if self.getPreviousObservation() is not None:
            if gameState.getAgentPosition(self.index) == self.start:
                return -100
        return 0

    def updateWeights(self, gameState: capture.GameState, action, nextState, reward):
        if not gameState.getAgentState(self.index).isPacman:
            pass
        else:
            ReflexCaptureAgent.updateWeights(self, gameState, action, nextState, reward)

    def getWeights(self, gameState: capture.GameState):
        return self.weights


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def registerInitialState(self, gameState: capture.GameState):
        self.weights = util.Counter(
            {'numInvaders': -1000,
             'onDefense': 100,
             'invaderDistance': -10,
             'stop': -100,
             'reverse': -2,
             'watchDist': -5,
             'invaderDanger': -20,
             'myScaredTimer': -2,
             'distFromPartnerIfGhost': 1,
             'invaderToClosestCapsule': 2
             })
        ReflexCaptureAgent.registerInitialState(self, gameState)

    def evaluate(self, gameState: capture.GameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState)
        return features * weights

    def getFeatures(self, gameState: capture.GameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        newPosition = myState.getPosition()

        # Feature 1
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Feature 2
        # Computes distance to invaders
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        # Feature 3
        if len(invaders) > 0:
            dists = [self.getMazeDistance(newPosition, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        # Feature 4
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        # Feature 5
        if action == rev:
            features['reverse'] = 1
        # Feature 6
        # nearest position to center
        watchPosition = util.nearestPoint((self.homeWidthBoundary, gameState.data.layout.height / 2))
        features['watchDist'] = self.getMazeDistance(newPosition, watchPosition)

        # Feature 7
        features['invaderDanger'] = features['invaderDistance'] * features['numInvaders']

        # Feature 8
        features['myScaredTimer'] = gameState.getAgentState(self.index).scaredTimer
        features['invaderDistance'] -= (2 * features['myScaredTimer'] * features['invaderDistance'])

        # Feature 9 -- so we can cover more ground
        partnerState = gameState.getAgentState(self.partnerIndex)
        partnerIsPacman = partnerState.isPacman
        if partnerIsPacman:
            features['distFromPartnerIfGhost'] = 0
        else:
            features['distFromPartnerIfGhost'] = self.getMazeDistance(partnerState.getPosition(), newPosition)

        # Feature 10
        features['invaderToClosestCapsule'] = self.invaderClosestToCapsule(successor)

        return features

    def invaderClosestToCapsule(self, successor:capture.GameState):
        """
        Computes the distance between the nearest invader and the closest capsule.
        """
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        capsules = self.getCapsules(successor)

        if not invaders or not capsules:
            return 0
        sd = sys.maxsize
        for i in invaders:
            for c in capsules:
                sd = min(sd, self.getMazeDistance(i.getPosition(), c))
        return sd

    def getReward(self, gameState:capture.GameState, action, nextState:capture.GameState):
        reward = 0
        curPos = gameState.getAgentPosition(self.index)
        if len(self.getFood(gameState).asList()) > len(self.getFood(nextState).asList()):
            reward -= 5
        if action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            next_position = self.getNextPosition(curPos, action)
            opps = self.getOpponents(nextState)

            oppsPositions = [nextState.getAgentPosition(opp) for opp in opps]
            if next_position in oppsPositions:
                return 20 # kill the opps
        return 0

    def getWeights(self, gameState: capture.GameState):
        return self.weights

