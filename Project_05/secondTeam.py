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


        # concerning team belonging and team territory
        for i in self.agentsOnTeam:
            if i != self.index:
                self.partnerIndex = i
        self.enemyAgentsIndices = self.getOpponents(gameState)

        if not self.red:  # if blue, the right half is ours
            self.homeTerritoryNoWalls = [(x, y) for (x, y) in self.allCoordsNoWalls if x >= self.layout.width // 2]
            self.oppTerritoryNoWalls = [(x, y) for (x, y) in self.allCoordsNoWalls if x < self.layout.width // 2]
        else:
            self.homeTerritoryNoWalls = [(x, y) for (x, y) in self.allCoordsNoWalls if x < self.layout.width // 2]
            self.oppTerritoryNoWalls = [(x, y) for (x, y) in self.allCoordsNoWalls if x >= self.layout.width // 2]

        # each member has a role
        # maybe these roles will switch during the game?
        # Comment them out for now

        # if self.index == self.agentsIndices[0]:
        #     self.iAmAttacker = True
        # else:
        #     self.iAmAttacker = False

        # bookkeeping and history-tracking
        self.actionHistory = []
        self.foodThreshold = 0  # hardcoded, will experiment with this; might find a way to make it depend on total food available and layout
        self.weights = util.Counter()
        self.features = util.Counter()

        # Q-learning parameters
        self.foodRewardMultiplier = 10 # might also make this dynamic
        self.discount = 1
        self.alpha = 0.1

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

        self.actionHistory.append(bestAction)
        successor = gameState.generateSuccessor(self.index, bestAction)
        reward = self.getReward(gameState)
        # if reward is None:
        #     print ("none")
        #     exit(1)
        self.updateWeights(gameState, bestAction, successor, reward)
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

    # Ported from Project 3
    def updateWeights(self, gameState: capture.GameState, action, nextState, reward):
        prevObs = self.getPreviousObservation()
        if prevObs is not None and prevObs.getAgentState(self.index) is not None:
            curObs = self.getCurrentObservation()

            difference = reward + (self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(
                curObs, action)

            curFeatures = self.getFeatures(curObs, action)
            curWeights = self.getWeights(curObs, action)

            if curFeatures is not None and curWeights is not None:
                for feature in curFeatures:
                    self.weights[feature] += self.alpha * difference * curWeights[feature]
                print(self.weights)
        return

    def evaluate(self, gameState: capture.GameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    # temporary

    def getFeatures(self, gameState: capture.GameState, action):
        """
        Returns a counter of features for the state
        """
        pass

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        pass

    def getReward(self, gameState):
        """
        To determine the external reward
        :param gameState:
        :return:
        """
        return 0

    # Ported from Project 3
    def getQValue(self, gameState: capture.GameState, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        # qvalue is the linear combination (or dot product) of weights and feature values.
        for feature in self.getFeatures(gameState, action):
            weight = self.getFeatures(gameState, action)[feature]
            qValue += weight * self.getFeatures(gameState, action)[feature]
        return qValue

    # Ported from Project 3
    # might have to use currentObservation since compute after leaving a state...not sure
    def computeValueFromQValues(self, gameState: capture.GameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            return 0.0
        q_value = - sys.maxsize

        for action in legal_actions:
            q_value = max(q_value, self.getQValue(gameState, action))
        return q_value


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
            'disToFood': -1,
            'foodInStomach': 50,
            'disToGhost': 0.75,
            'disToPartner': -5,
            'disToCap': -10,
            'returnHome': -50,
            'isDeadEnd': 10
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

        # Feature 2: Distance To The Nearest Capsule

        capsulePositions = self.getCapsules(gameState)
        if len(capsulePositions) > 0:  # I guess we'll double-check everything now
            minCapDistance = min([self.getMazeDistance(newPosition, capsule) for capsule in capsulePositions])
            features['disToCap'] = minCapDistance

        # Feature 3: Distance To The Nearest Ghost
        opps = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        # we can't read the far ones so case for None
        oppGhosts = [opp for opp in opps if not opp.isPacman and opp.getPosition() is not None]
        if len(oppGhosts) > 0 and gameState.getAgentState(self.index).isPacman:
            oppGhostPositions = [ghost.getPosition() for ghost in oppGhosts]
            distsToGhosts = [self.getMazeDistance(newPosition, ghostPos) for ghostPos in oppGhostPositions]
            closestGhostDist = min(distsToGhosts)
            ghostScaredTimer = max([ghost.scaredTimer for ghost in oppGhosts])

            if ghostScaredTimer < 4:
                closestGhostDist *= -1
            features['disToGhost'] = closestGhostDist

        # Feature 4: Distance To Teammate:  Admire, you wanted this one; figure out how to use it.
        if self.partnerIndex is not None:  # should always return true as well
            partnerPosition = gameState.getAgentState(self.partnerIndex).getPosition()
            disToPartner = self.getMazeDistance(newPosition, partnerPosition)
            features['disToPartner'] = disToPartner

        # Feature 5: from baseline
        features['successorScore'] = -len(foodList)

        # Feature 6: there was a variable all along lol
        foodInStomach = gameState.getAgentState(self.index).numCarrying
        features['foodInStomach'] = foodInStomach

        # Feature 7: closely related to 6
        if foodInStomach > self.foodThreshold:
            features['returnHome'] = self.getMazeDistance(newPosition, self.start)
            # distance to home was messing up because it would be decreasing as we went home

        # Feature 8: Avoid dead ends
        legalActionsInSuccessor = successor.getLegalActions(self.index)
        if len(legalActionsInSuccessor) == 2:
            features['isDeadEnd'] = 1
        return features

    def getReward(self, gameState: capture.GameState):
        return self.foodReward(gameState) + self.wasEatenReward(gameState)

    def foodReward(self, gameState: capture.GameState):
        if self.getCurrentObservation() is not None:
            numPrevFood = len(self.getFood(self.getCurrentObservation()).asList())
            numCurFood = len(self.getFood(gameState).asList())
            eaten = numCurFood - numPrevFood
            # print(eaten)
            return eaten * self.foodRewardMultiplier
        return 0

    def wasEatenReward(self, gameState: capture.GameState):
        if self.getCurrentObservation().getAgentState(self.index).isPacman \
                and gameState.getAgentPosition(self.index) == self.start:
            return -100
        return 0

    def getWeights(self, gameState: capture.GameState, action):
        return self.weights


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def evaluate(self, gameState: capture.GameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

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
        return self.weights
        # return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
