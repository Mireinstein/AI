# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostStates.sort(key=lambda x: manhattanDistance(newPos, x.getPosition()))
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        food_locations = newFood.asList()
        food_locations.sort(key=lambda x: manhattanDistance(x, newPos))
        ghost_locations = [g.getPosition() for g in newGhostStates]

        # if ghosts are close and are not scared, run away
        for ghost_location in ghost_locations:
            if manhattanDistance(ghost_location, newPos) < 2 and newScaredTimes[0] > 0:
                return newScaredTimes[0] / 10000
            if manhattanDistance(ghost_location, newPos) < 2 and newScaredTimes[0] == 0:
                return -sys.maxsize / 10000

        num_food_left = successorGameState.getNumFood()

        # State is bad if pacman is in danger
        if ((manhattanDistance(currentGameState.getPacmanPosition(), newPos)) == 0
                and manhattanDistance(newPos,newGhostStates[0].getPosition()) >= 2):
            return -sys.maxsize / 10000

        # reward the behaviour of finishing food
        if num_food_left == 0:
            return sys.maxsize / 10000

        # incorporate distance to closest food
        return (10000 / num_food_left) - (manhattanDistance(newPos, food_locations[0]) / 10000)

        # return (1/dist_to_closest_food)

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        num_of_agents = gameState.getNumAgents()

        def maxValue(agentIndex, gameState, depth):
            # Take no action if we've reached the goal; just return the urility
            if depth == 0 or (gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState), None

            value = -sys.maxsize - 1
            move = None
            # Choose the action that results in the best utility(Take the max of the minimums)
            for action in gameState.getLegalActions(agentIndex):
                current_value, current_move = minValue((agentIndex + 1) % num_of_agents,
                                                       gameState.generateSuccessor(agentIndex, action), depth)

                # Take the action that results in the maximum utility
                if current_value > value:
                    value, move = current_value, action
            return value, move

        def minValue(agentIndex, gameState, depth):
            if depth == 0 or (gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState), None

            value = sys.maxsize
            move = None
            for action in gameState.getLegalActions(agentIndex):
                # if it's not the last ghost's turn, don't update depth
                if agentIndex != num_of_agents - 1:
                    current_value, current_move = minValue((agentIndex + 1) % num_of_agents,
                                                           gameState.generateSuccessor(agentIndex, action), depth)

                # update depth since it's Pacman's turn to move
                else:
                    current_value, current_move = maxValue((agentIndex + 1) % num_of_agents,
                                                           gameState.generateSuccessor(agentIndex, action), depth - 1)

                # Take the action that results in the least utility
                if current_value < value:
                    value, move = current_value, action
            return value, move

        utility, action = maxValue(0, gameState, self.depth)

        return action

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # a list of tuples containing info about the first level actions (score, action).
        firstLevelList = []

        num_of_agents = gameState.getNumAgents()

        def maxValue(agentIndex, gstate, depth, alpha, beta):
            if depth == 0 or (gstate.isWin() or gstate.isLose()):
                return self.evaluationFunction(gstate)

            v = -sys.maxsize - 1
            legal_acts = gstate.getLegalActions(agentIndex)
            for a in legal_acts:
                v = max(v, minValue((agentIndex + 1) % num_of_agents, gstate.generateSuccessor(agentIndex, a), depth,
                                    alpha, beta))
                # only record to our list if we are at the first level (before we have modified depth).
                if depth == self.depth: firstLevelList.append((v, a))

                if v > beta:
                    return v
                alpha = max(alpha, v)

            return v

        def minValue(agentIndex, gstate, depth, alpha, beta):
            if depth == 0 or (gstate.isWin() or gstate.isLose()):
                return self.evaluationFunction(gstate)

            v = sys.maxsize - 1
            legal_acts = gstate.getLegalActions(agentIndex)
            for a in legal_acts:
                if agentIndex != num_of_agents - 1:
                    v = min(v,
                            minValue((agentIndex + 1) % num_of_agents, gstate.generateSuccessor(agentIndex, a), depth,
                                     alpha, beta))
                else:
                    v = min(v, maxValue((agentIndex + 1) % num_of_agents, gstate.generateSuccessor(agentIndex, a),
                                        depth - 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        utility = maxValue(0, gameState, self.depth, -sys.maxsize - 1, sys.maxsize)

        # query out list for where the action is optimal
        for tuple in firstLevelList:
            if tuple[0] == utility:
                return tuple[1]
        return None
        "My code ends here"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        num_of_agents = gameState.getNumAgents()

        def maxValue(agentIndex, gameState, depth):
            # Take no action if we've reached the goal; just return the utility
            if depth == 0 or (gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState), None

            value = -sys.maxsize - 1
            move = None
            # Choose the action that results in the best utility(Take the max of the minimums)
            for action in gameState.getLegalActions(agentIndex):
                current_value, current_move = expectedScore((agentIndex + 1) % num_of_agents,
                                                            gameState.generateSuccessor(agentIndex, action), depth)

                # Take the action that results in the maximum utility
                if current_value > value:
                    value, move = current_value, action
            return value, move

        def expectedScore(agentIndex, gameState, depth):
            # Take no action if we've reached the goal; just return the utility
            if gameState.isWin() or gameState.isLose() or depth < 1:
                return self.evaluationFunction(gameState), None

            legal_actions = gameState.getLegalActions(agentIndex)

            sum_scores = 0

            for action in legal_actions:
                if agentIndex != num_of_agents - 1:
                    current_value, current_move = expectedScore((agentIndex + 1) % num_of_agents,
                                                                gameState.generateSuccessor(agentIndex, action), depth)
                    sum_scores += current_value

                # update depth since it's Pacman's turn to move
                else:
                    current_value, current_move = maxValue((agentIndex + 1) % num_of_agents,
                                                           gameState.generateSuccessor(agentIndex, action), depth - 1)
                    sum_scores += current_value

            # Calculate the average score and select a random action
            avg_score = sum_scores / len(legal_actions)
            action = legal_actions[random.randrange(0, len(legal_actions))]
            return avg_score, action

        return maxValue(0, gameState, self.depth)[1]

        "** our code ends here **"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    Our evaluation function is a linear combination of the reciprocals of the distance to the closest ghost,
    the distance to the closest food, and the number of foods left. Each reciprocal is weighted differently,
    based on how typically those values are in a Pacman game.

    There are also the edge case when the state being evaluated has no foods.

    There are also general 'wisdoms' of Pacman: - if you are close to an un-scared ghost, it's bad.
                                                - if ghosts are currently scared do your best to eat them

    Our weights reflect the importance of each measure but also the ranges in which they lie in the game.
            * The amount of food remaining is weighed heaviest,
            * Followed by the urgency to eat a ghost for the short duration they are scared
            * and last, the distance to the closest pellet (In general, you can eat the closest food anytime you are not in danger)


    """
    "*** YOUR CODE HERE ***"
    score = 0
    pacman_position = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newGhostStates.sort(key=lambda x: manhattanDistance(pacman_position, x.getPosition()))
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    food_locations = newFood.asList()
    food_locations.sort(key=lambda x: manhattanDistance(x, pacman_position), reverse=True)
    food_distances = [manhattanDistance(pacman_position, location) for location in food_locations]

    ghost_locations = [g.getPosition() for g in newGhostStates]
    ghost_distances = [manhattanDistance(pacman_position, location) for location in ghost_locations]

    num_food_left = currentGameState.getNumFood()

    if num_food_left == 0:
        return sys.maxsize

    if min(ghost_distances) <= 1 and max(newScaredTimes) == 0:
        return -sys.maxsize - 1

    score += 1000 * (1 / num_food_left)

    if newScaredTimes[0] > 0:
            score += 100 * (1/ manhattanDistance(pacman_position, ghost_locations[0]))

    score += 1/min(food_distances)

    return score

    "** Our code ends here"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
