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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodList = newFood.asList()
        newGhostPos = successorGameState.getGhostPositions()

        min_dist_to_food = 10000000
        for food in foodList:
            min_dist_to_food = min(min_dist_to_food, manhattanDistance(newPos, food))

        min_dist_to_ghost = 10000000
        for ghost in newGhostPos:
            min_dist_to_ghost = min(min_dist_to_ghost, manhattanDistance(newPos, ghost))

        eval = (0.4) * successorGameState.getScore() + (0.3) * (1 / min_dist_to_food)
        if min_dist_to_ghost < 3:
            eval += (0.4) * min_dist_to_ghost

        #return successorGameState.getScore()
        return eval

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def max_value(self, gameState, depth):      #note that Pacman is always the maximizer so the agentIndex = 0
        if gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        v = [-float("inf"), None]       #tracks the value of the max valued action, and the action itself
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = self.min_value(successor, depth, 1)
            #eval = val[0]
            if val[0] > v[0]:
                v[0] = val[0]
                v[1] = action
        return tuple(v)

    def min_value(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        v = [float("inf"), None]
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            num_agents = gameState.getNumAgents()
            if agent == num_agents - 1:
                if depth == self.depth:
                    eval = self.evaluationFunction(successor)
                    if eval < v[0]:
                        v[0] = eval
                        v[1] = action
                else:
                    val = self.max_value(successor, depth + 1)
                    if val[0] < v[0]:
                        v[0] = val[0]
                        v[1] = action
            else:
                val = self.min_value(successor, depth, agent + 1)
                if val[0] < v[0]:
                    v[0] = val[0]
                    v[1] = action
        return tuple(v)

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
        return self.max_value(gameState, 1)[1]

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value(self, gameState, depth, alpha, beta):      #note that Pacman is always the maximizer so the agentIndex = 0
        if gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        v = [-float("inf"), None]       #tracks the value of the max valued action, and the action itself
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = self.min_value(successor, depth, 1, alpha, beta)
            #eval = val[0]
            if val[0] > v[0]:
                v[0] = val[0]
                v[1] = action
            if v[0] > beta:
                return tuple(v)
            alpha = max(alpha, v[0])
        return tuple(v)

    def min_value(self, gameState, depth, agent, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        v = [float("inf"), None]
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            num_agents = gameState.getNumAgents()
            if agent == num_agents - 1:
                if depth == self.depth:
                    eval = self.evaluationFunction(successor)
                    if eval < v[0]:
                        v[0] = eval
                        v[1] = action
                else:
                    val = self.max_value(successor, depth + 1, alpha, beta)
                    if val[0] < v[0]:
                        v[0] = val[0]
                        v[1] = action
            else:
                val = self.min_value(successor, depth, agent + 1, alpha, beta)
                if val[0] < v[0]:
                    v[0] = val[0]
                    v[1] = action
            if v[0] < alpha:
                return tuple(v)
            beta = min(beta, v[0])
        return tuple(v)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 1, -float("inf"), float("inf"))[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_value(self, gameState, depth):      #note that Pacman is always the maximizer so the agentIndex = 0
        if gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        v = [-float("inf"), None]       #tracks the value of the max valued action, and the action itself
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = self.exp_value(successor, depth, 1)
            #eval = val[0]
            if val[0] > v[0]:
                v[0] = val[0]
                v[1] = action
        return tuple(v)

    def exp_value(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        v = [float("inf"), None]
        sum = 0
        num_actions = len(gameState.getLegalActions(agent))
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            num_agents = gameState.getNumAgents()
            if agent == num_agents - 1:
                if depth == self.depth:
                    eval = self.evaluationFunction(successor)
                    sum += eval
                else:
                    val = self.max_value(successor, depth + 1)
                    sum += val[0]
            else:
                val = self.exp_value(successor, depth, agent + 1)
                sum += val[0]
        v[0] = (1.0 / num_actions) * sum
        # note: we dont need to track the actions of the adversaries
        return tuple(v)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 1)[1]
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I did some stuff.
    """
    "*** YOUR CODE HERE ***"
    #things to condiser: current score(pellets already eaten), distance to nearest food,
    # distance to nearest ghost, if ghost is really close fuck the food


    pos = currentGameState.getPacmanPosition()
    food_pos = (currentGameState.getFood()).asList()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghost_state.scaredTimer for ghost_state in ghost_states]
    ghost_pos = currentGameState.getGhostPositions()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    ws, wf, wg, wsg, wst = [10.0, 2.0, -5.0, 8.0, 1.0]

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return -float("inf")

    min_dist_to_food = float("inf")
    min_dist_to_ghost = float("inf")
    min_dist_to_capsule = float("inf")

    food = 0
    for f in food_pos:
        min_dist_to_food = min(min_dist_to_food, manhattanDistance(pos, f))
        food = 1.0 / (min_dist_to_food)
        if min_dist_to_food == 1:
            wf = 10.0

    ghost = 0
    for gp in ghost_pos:
        min_dist_to_ghost = min(min_dist_to_ghost, manhattanDistance(pos, gp))
        if min_dist_to_ghost < 5:
            wg = -8.0
        ghost = 1.0 / min_dist_to_ghost


    scared = 0
    if scared_times[0] > 0:
        min_dist_to_sghost = float("inf")
        for gp in ghost_pos:
            min_dist_to_sghost = min(min_dist_to_sghost, manhattanDistance(pos, gp))
        scared = 1.0 / min_dist_to_sghost

    scared_time = scared_times[0] * (1.0 / min_dist_to_ghost)

    evaluation = ws * score + wf * food + wg * ghost + wsg * scared + wst * scared_time

    return evaluation

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
