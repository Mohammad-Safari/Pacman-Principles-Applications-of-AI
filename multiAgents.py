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
import random
import util

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]
        newCapsules = successorGameState.getCapsules()

        # the last distance to the closest food
        lastClosestFoodDist = min(manhattanDistance(newPos, food) for food in currentGameState.getFood().asList()) \
            if not len(newFood.asList()) == 0 else 999
        # the distance to the closest food
        closestFoodDist = min(manhattanDistance(newPos, food) for food in newFood.asList()) \
            if not len(newFood.asList()) == 0 else 999
        # the distance to the closest ghost
        closestGhostDist = min(manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates) \
            if not len(newGhostStates) == 0 else 999

        maxNegPoint = -200
        maxPosPoint = 100
        nearDanger = 3
        isScary = False

        # if ghost is not near, no need to worry a lot;
        # but if it is near, then it is scary and can cost life
        # 1)
        # ghostDistEval = 0 if (closestGhostDist > 2*nearDanger) else \
        #     maxNegPoint/closestGhostDist if (closestGhostDist > 2*nearDanger) else maxNegPoint
        # 2)
        ghostDistEval = 0 if (closestGhostDist > nearDanger) else maxNegPoint

        # *) if ghost is not near, try to eat food really hard;
        # but if it is near, then dont try hard as it costs life
        # *) if action cause eating dots is greate and if it take us near to dots,
        # still good else if getting distance from dot, with the more distance,
        # the lower evaluation score it gets
        # *) also we prefer a range of safe moves instead of stopping
        foodDistEval = maxPosPoint if (currentGameState.getNumFood() > successorGameState.getNumFood()) else \
            maxPosPoint/2 if (lastClosestFoodDist > closestFoodDist) else maxPosPoint/closestFoodDist \
            + 0 if action != 'Stop' else random.randrange(-nearDanger*2, nearDanger*2)

        # score change in environment is important like eating near ghost in scary mode
        diffScore = successorGameState.getScore() - currentGameState.getScore()

        finalEval = ghostDistEval * \
            (-1 if isScary else 1) + foodDistEval + diffScore
        return finalEval


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
        agents = gameState.getNumAgents()
        lastGhost = agents-1
        maxScore = 10**10

        def minimaxPacman(state=gameState, depth=1):
            # before pacmans turns, it can win or lose
            # base condition of problem division is reaching beyond depth
            if(state.isWin() or state.isLose() or depth > self.depth):
                return self.evaluationFunction(state)

            # expanding tree of pacman choices and choosing
            # max value in combination of subproblems
            legalMoves = state.getLegalActions(0)
            previousScore, previousMove = -maxScore, None
            for move in legalMoves:
                score = minimaxGhost(
                    state.generateSuccessor(0, move), depth, 1)
                if (score > previousScore):
                    previousScore, previousMove = score, move

            if (depth == 1):  # on first call(top depth) return action instead of score
                return previousMove
            return previousScore

        def minimaxGhost(state, depth, agent):
            # pacman could have win or lose in this turn by preceding move
            if (state.isWin() or state.isLose()):
                return self.evaluationFunction(state)

            # expanding tree for ghost agent moves
            legalMoves = state.getLegalActions(agent)
            previousScore, previousMove = maxScore, None
            for move in legalMoves:
                if (agent == lastGhost):
                    # this is last ghost move and pacman should move next turn
                    score = minimaxPacman(
                        state.generateSuccessor(agent, move), depth+1)
                    if (score < previousScore):
                        previousScore, previousMove = score, move
                else:
                    score = minimaxGhost(state.generateSuccessor(
                        agent, move), depth, agent+1)
                    if (score < previousScore):
                        previousScore, previousMove = score, move
            # min score of ghost is returned
            return previousScore

        return minimaxPacman()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        agents = gameState.getNumAgents()
        lastGhost = agents-1
        maxScore = 10**10
        # alpha is meant to be the best hapenning score of the maximizing player
        # beta is meant to be the worst hapenning score of the minimizing player

        def minimaxPacman(state=gameState, depth=1, alpha=-maxScore, beta=maxScore):
            # before pacmans turns, it can win or lose
            # base condition of problem division is reaching beyond depth
            if(state.isWin() or state.isLose() or depth > self.depth):
                return self.evaluationFunction(state)

            # expanding tree of pacman choices and choosing
            # max value in combination of subproblems
            legalMoves = state.getLegalActions(0)
            previousScore, previousMove = -maxScore, None
            for move in legalMoves:
                score = minimaxGhost(
                    state.generateSuccessor(0, move), depth, alpha, beta, 1)
                if (score > previousScore):
                    previousScore, previousMove = score, move
                # prunning
                # set alpha to max score of pacman
                alpha = previousScore if previousScore > alpha else alpha
                # if the current best score of pacman is already more than beta(least score from parent minimizer), prune
                # because no way maximizer could send lower than beta to minimizer to minimize ghost score
                # this cannot effect parent minimum
                if(alpha > beta):
                    break

            return previousMove if depth == 1 else previousScore

        def minimaxGhost(state, depth, alpha, beta, agent):
            # pacman could have win or lose in this turn by preceding move
            if (state.isWin() or state.isLose()):
                return self.evaluationFunction(state)

            # expanding tree for ghost agent moves
            legalMoves = state.getLegalActions(agent)
            previousScore, previousMove = maxScore, None
            for move in legalMoves:
                if (agent == lastGhost):
                    # this is last ghost move and pacman should move next turn
                    score = minimaxPacman(
                        state.generateSuccessor(agent, move), depth+1, alpha, beta)
                    if (score < previousScore):
                        previousScore, previousMove = score, move
                else:
                    score = minimaxGhost(state.generateSuccessor(
                        agent, move), depth, alpha, beta, agent+1)
                    if (score < previousScore):
                        previousScore, previousMove = score, move
                # prunning
                # set beta min score of ghost
                beta = previousScore if previousScore < beta else beta
                # if the current least score of ghost is already lesser than alpha(most score from parent maximizer), prune
                # because no way minimizer could send more than beta to maximizer to minimize ghost score
                # this cannot not effect parent maximum
                if(alpha > beta):
                    break
            # min score of ghost is returned
            return previousScore

        return minimaxPacman()


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
        agents = gameState.getNumAgents()
        lastGhost = agents-1
        maxScore = 10**10

        def minimaxPacman(state=gameState, depth=1):
            # before pacmans turns, it can win or lose
            # base condition of problem division is reaching beyond depth
            if(state.isWin() or state.isLose() or depth > self.depth):
                return self.evaluationFunction(state)

            # expanding tree of pacman choices and choosing
            # max value in combination of subproblems
            legalMoves = state.getLegalActions(0)
            previousScore, previousMove = -maxScore, None
            for move in legalMoves:
                score = minimaxGhost(
                    state.generateSuccessor(0, move), depth, 1)
                if (score > previousScore):
                    previousScore, previousMove = score, move

            if (depth == 1):  # on first call(top depth) return action instead of score
                return previousMove
            return previousScore

        def minimaxGhost(state, depth, agent):
            # pacman could have win or lose in this turn by preceding move
            if (state.isWin() or state.isLose()):
                return self.evaluationFunction(state)

            # expanding tree for ghost agent moves
            legalMoves = state.getLegalActions(agent)
            previousScore = 0
            for move in legalMoves:
                if (agent == lastGhost):
                    # this is last ghost move and pacman should move next turn
                    previousScore += minimaxPacman(
                        state.generateSuccessor(agent, move), depth+1)
                else:
                    previousScore += minimaxGhost(state.generateSuccessor(
                        agent, move), depth, agent+1)
            # min score of ghost is returned
            return previousScore/len(legalMoves)

        return minimaxPacman()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
    curCapsules = currentGameState.getCapsules()

    mostDist = 999
    maxNegPoint = -2e5
    maxPosPoint = 1e5
    nearDanger = 2
    # if pacman can eat ghost(very risky param beacuse it take pacman near ghost with much weight)
    isScary = False if curScaredTimes[0] == 0 else True
    


    # the distance to the closest food
    foodDist = [manhattanDistance(curPos, food) for food in curFood.asList()]
    minFoodDist = min(foodDist) if len(foodDist) > 0 else mostDist
    # the distance to the closest ghost
    ghostDist = [manhattanDistance(curPos, ghost.getPosition()) for ghost in curGhostStates]
    minGhostDist = min(ghostDist) if len(ghostDist) > 0 else 0
    # the distance to the closest capsule
    capsuleDist = [manhattanDistance(curPos, capsule) for capsule in curCapsules]
    minCapsuleDist = min(capsuleDist) if len(capsuleDist) > 0 else mostDist
    

    # if ghost is not near, no need to worry a lot;
    # but if it is near, then it is scary and can cost life
    ghostDistEval = 0 if (minGhostDist > nearDanger) else maxNegPoint

    # *) if ghost is not near, try to eat food really hard;
    # but if it is near, then dont try hard as it costs life
    # *) if action cause eating dots is greate and if it take us near to dots,
    # still good else if getting distance from dot, with the more distance,
    # the lower evaluation score it gets
    # *) also we prefer a range of safe moves instead of stopping
    foodDistEval =  (maxPosPoint/2)/(len(foodDist)+1) + (maxPosPoint/2)/minFoodDist if minGhostDist > nearDanger else 0

    # 
    capsuleDistEval = len(capsuleDist) + minCapsuleDist if minGhostDist < nearDanger else 0

    finalEval = ghostDistEval * \
        (-1 if isScary else 1) + foodDistEval + capsuleDistEval + currentGameState.getScore()*maxPosPoint/2
    return finalEval


# Abbreviation
better = betterEvaluationFunction
