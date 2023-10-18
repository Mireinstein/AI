# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        currentValues=self.values.copy()
        iterations=self.iterations
        while True:
            self.values=currentValues.copy()
            # delta=0
            iterations-=1
            for state in self.mdp.getStates():
                utility=0
                for action in self.mdp.getPossibleActions(state):
                    state_prob_list = self.mdp.getTransitionStatesAndProbs(state, action)
                    q_value = 0
                    for state_prob_tuple in state_prob_list:
                        prob_to_next_state = state_prob_tuple[1]
                        next_state = state_prob_tuple[0]
                        reward_to_next_state = self.mdp.getReward(state, action, next_state)
                        q_value += prob_to_next_state * (
                                    reward_to_next_state + (self.discount * self.values[next_state]))
                    if q_value>utility:
                        utility=q_value
                currentValues[state]=utility

                # if abs(currentValues[state]-self.values[state])>delta:
                #     delta=abs(currentValues[state]-self.values[state])

            # if delta<=0.05*(1-self.discount)/self.discount:
            #     # self.values=currentValues.copy()
            #     break
            if iterations<0:
                break



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        state_prob_list=self.mdp.getTransitionStatesAndProbs(state,action)
        q_value=0
        for state_prob_tuple in state_prob_list:
            prob_to_next_state=state_prob_tuple[1]
            next_state=state_prob_tuple[0]
            reward_to_next_state=self.mdp.getReward(state,action,next_state)
            q_value+=prob_to_next_state*(reward_to_next_state+(self.discount*self.getValue(next_state)))

        return q_value

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get legal actions from this state
        legal_actions=self.mdp.getPossibleActions(state)

        # create a dictionary of action to reward
        action_reward_dict=util.Counter()

        for action in legal_actions:
            action_reward_dict[action]=self.computeQValueFromValues(state,action)

        # Return the action with the highest Q-value
        return action_reward_dict.argMax()


        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
