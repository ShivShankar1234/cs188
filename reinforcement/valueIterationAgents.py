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
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            vals = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                actions = self.mdp.getPossibleActions(state)
                q_vals = {}
                for action in actions:
                    q_vals[action] = self.computeQValueFromValues(state, action)
                vals[state] = max(q_vals.values())
            self.values = vals



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
        max_util = float("-inf")
        max_util_action = None
        if self.mdp.isTerminal(state):
            return None

        transition_states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        curr_util = 0
        for next_state, prob in transition_states_and_probs:
            curr_util += prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.getValue(next_state))
        return curr_util

        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_util = -float("inf")
        max_util_action = None

        actions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state) or not actions:
            return None
        for action in actions:
            curr_q_val = self.computeQValueFromValues(state, action)
            if curr_q_val > max_util:
                max_util_action = action
                max_util = curr_q_val
        return max_util_action

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        i = 0
        while i < self.iterations:
            for state in self.mdp.getStates():
                q_vals = util.Counter()         #using a counter(dictionary) lets us wrap around
                for action in self.mdp.getPossibleActions(state):
                    q_vals[action] = self.getQValue(state, action)
                self.values[state] = q_vals[q_vals.argMax()]
                i += 1
                if i >= self.iterations:
                    return



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        preds = {}
        states = self.mdp.getStates()
        for s in states:
            preds[s] = set()

        pq = util.PriorityQueue()

        for s in states:        #simultaneously filling preds pq
            q_vals_s = util.Counter()
            for action in self.mdp.getPossibleActions(s):
                successors_and_probs = self.mdp.getTransitionStatesAndProbs(s, action)
                for next_state, prob in successors_and_probs:
                    if prob != 0:
                        preds[next_state].add(s)
                q_vals_s[action] = self.getQValue(s, action)

            if not self.mdp.isTerminal(s):
                diff = abs(self.getValue(s) - q_vals_s[q_vals_s.argMax()])
                pq.update(s, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                return

            s = pq.pop()
            if not self.mdp.isTerminal(s):
                q_vals_s = util.Counter()
                for action in self.mdp.getPossibleActions(s):
                    q_vals_s[action] = self.getQValue(s, action)
                self.values[s] = q_vals_s[q_vals_s.argMax()]

            for p in preds[s]:
                q_vals_p = util.Counter()
                for action in self.mdp.getPossibleActions(p):
                    q_vals_p[action] = self.getQValue(p, action)
                diff = abs(self.getValue(p) - q_vals_p[q_vals_p.argMax()])
                if diff > self.theta:
                    pq.update(p, -diff)

