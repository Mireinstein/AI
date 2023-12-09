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

    from captureAgents import CaptureAgent
    from game import Directions
    import random

    class MyPacmanAgent(CaptureAgent):
        def chooseAction(self, gameState):
            """
            Chooses an action based on the known positions of opponent ghosts.
            """
            opponent_ghosts = self.getOpponentGhosts(gameState)

            if not opponent_ghosts:
                # No opponent ghosts are visible, choose a random action
                legal_actions = gameState.getLegalActions(self.index)
                return random.choice(legal_actions)

            # Choose an action based on known opponent ghost positions
            return self.chooseActionBasedOnOpponentPositions(gameState, opponent_ghosts)

        def getOpponentGhosts(self, gameState):
            """
            Returns the positions of visible opponent ghosts.
            """
            opponent_ghosts = []
            for opponent_index in self.getOpponents(gameState):
                opponent_state = gameState.getAgentState(opponent_index)
                if opponent_state.isGhost and opponent_state.getPosition() is not None:
                    opponent_ghosts.append(opponent_state.getPosition())
            return opponent_ghosts

        def chooseActionBasedOnOpponentPositions(self, gameState, opponent_ghosts):
            """
            Chooses an action based on the known positions of opponent ghosts.
            """
            my_position = gameState.getAgentPosition(self.index)

            # Example: Directly avoid ghosts if they are nearby
            for ghost_position in opponent_ghosts:
                if self.getMazeDistance(my_position, ghost_position) <= 3:
                    return self.avoidGhostAction(gameState, ghost_position)

            # Example: If no ghosts are nearby, choose a random action
            legal_actions = gameState.getLegalActions(self.index)
            return random.choice(legal_actions)

        def avoidGhostAction(self, gameState, ghost_position):
            """
            Chooses an action to avoid a ghost based on its position.
            """
            legal_actions = gameState.getLegalActions(self.index)
            my_position = gameState.getAgentPosition(self.index)

            # Example: Choose an action that moves away from the ghost
            best_action = None
            best_distance = float('inf')

            for action in legal_actions:
                successor = gameState.generateSuccessor(self.index, action)
                successor_position = successor.getAgentPosition(self.index)
                distance = self.getMazeDistance(successor_position, ghost_position)

                if distance < best_distance:
                    best_distance = distance
                    best_action = action

            return best_action


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
