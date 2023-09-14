# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # inititialize starting problem with cost of zero and game initial state
    node = (problem.getStartState(), None, 0, None);
    frontier = util.Stack()
    frontier.push(node)
    # A set containing visited states
    explored = set()
    # The path we will return
    path = []
    # Keep looking for the goal state
    while not problem.isGoalState(node[0]):
        if frontier.isEmpty():
            # return failure
            return []
        node = frontier.pop()
        # Path.append(node[1]);
        if problem.isGoalState(node[0]):
            # Backtrack to get the solution
            # append last action
            path.append(node[1])
            # as long as action is not none.
            while node[3] is not None:
                parent = node[3]
                action = parent[1]
                # Don't take None as an action.
                if action is not None:
                    path.append(action)
                node = parent
            # return the path
            path.reverse()
            return path
        # if we already visited this state, don't visit it again.
        if node[0] not in explored:
            # Add a visited state to the explored set.
            explored.add(node[0])
            # Add all children to the frontier
            for child in problem.getSuccessors(node[0]):
                # Add the quadruple representing (successor, action, cost, Parent)
                frontier.push((child[0], child[1], child[2], node))
    return path
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    node = (problem.getStartState(), None, 0, None);
    frontier = util.Queue()
    frontier.push(node)
    # A set containing visited states
    explored = set()
    # The path we will return
    path = []
    # Keep looking for the goal state
    while not problem.isGoalState(node[0]):
        if frontier.isEmpty():
            # return failure
            return []
        node = frontier.pop()
        # Path.append(node[1]);
        if problem.isGoalState(node[0]):
            # Backtrack to get the solution
            # append last action
            path.append(node[1])
            # as long as action is not none.
            while node[3] is not None:
                parent = node[3]
                action = parent[1]
                # Don't take None as an action.
                if action is not None:
                    path.append(action)
                node = parent
            # return the path
            path.reverse()
            return path
        # if we already visited this state, don't visit it again.
        if node[0] not in explored:
            # Add a visited state to the explored set.
            explored.add(node[0])
            # Add all children to the frontier
            for child in problem.getSuccessors(node[0]):
                # Add the quadruple representing (successor, action, cost, Parent)
                frontier.push((child[0], child[1], child[2], node))
    return path
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Quintuple : our old info, and fifth element = path cost
    node = (problem.getStartState(), None, 0, None, 0)
    frontier = util.PriorityQueue()
    frontier.push(node, node[4])
    # A set containing visited states
    explored = set()
    # The path we will return
    path = []
    # Keep looking for the goal state
    while not problem.isGoalState(node[0]):
        if frontier.isEmpty():
            # return failure
            return []
        node = frontier.pop()
        # Path.append(node[1]);
        if problem.isGoalState(node[0]):
            # Backtrack to get the solution
            # append last action
            path.append(node[1])
            # as long as action is not none.
            while node[3] is not None:
                parent = node[3]
                action = parent[1]
                # Don't take None as an action.
                if action is not None:
                    path.append(action)
                node = parent
            # return the path
            path.reverse()
            return path
        # if we already visited this state, don't visit it again.
        if node[0] not in explored:
            # Add a visited state to the explored set.
            explored.add(node[0])
            # Add all children to the frontier
            for child in problem.getSuccessors(node[0]):
                # calculate the cumulative path cost by adding previous node's path cost to child's individual cost
                path_cost = node[4] + child[2]
                # Add the quintuple representing (successor, action, cost, Parent, and path_cost)
                frontier.push((child[0], child[1], child[2], node, path_cost), path_cost)
    return path

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    node = (problem.getStartState(), None, 0, None, 0)
    # priority is path cost + heuristic(node)
    frontier = util.PriorityQueue()

    #accounting for the heuristic
    frontier.push(node, node[4] + heuristic(node[0], problem))
    # A set containing visited states
    explored = set()
    # The path we will return
    path = []
    # Keep looking for the goal state
    while not problem.isGoalState(node[0]):
        if frontier.isEmpty():
            # return failure
            return []
        node = frontier.pop()
        # Path.append(node[1]);
        if problem.isGoalState(node[0]):
            # Backtrack to get the solution
            # append last action
            path.append(node[1])
            # as long as action is not none.
            while node[3] is not None:
                parent = node[3]
                action = parent[1]
                # Don't take None as an action.
                if action is not None:
                    path.append(action)
                node = parent
            # return the path
            path.reverse()
            return path
        # if we already visited this state, don't visit it again.
        if node[0] not in explored:
            # Add a visited state to the explored set.
            explored.add(node[0])
            # Add all children to the frontier
            for child in problem.getSuccessors(node[0]):
                # calculate the cumulative path cost by adding previous node's path cost to child's individual cost
                path_cost = node[4] + child[2]
                # Add the quintuple representing (successor, action, cost, Parent, and path_cost)
                # NOTE : priority != pathcost ; we reevaluate the value of the heuristic at every child but we don't want it to stick.
                frontier.push((child[0], child[1], child[2], node, path_cost), path_cost + heuristic(child[0], problem))
    return path


    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
