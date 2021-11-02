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
    return  [s, s, w, s, w, w, s, w]

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
    # environment
    isGoalState = problem.isGoalState
    # fn: (next state/location, action, cost)
    getSuccessors = problem.getSuccessors
    currentLocation = problem.getStartState()  # (x, y)

    # our data structure needed for dfs
    visited = []  # visited location list
    directionSequence = util.Stack()  # will register path directions

    def depthFirstTraverse(successor):

        # expanding given node
        currentLocation, direction, cost = successor
        directionSequence.push(direction)
        visited.append(currentLocation)

        # checking if visiting goal state
        if(isGoalState(currentLocation)):
            return True

        # creating adjacent nodes
        availableSuccessors = getSuccessors(currentLocation)
        for successor in availableSuccessors:
            if(successor[0] not in visited):
                if (depthFirstTraverse(successor)):
                    return True

        # clearing direction to unrelevant nodes(of unsuccessful path)
        directionSequence.pop()
        return False

    # expanding root(initial) node
    visited.append(currentLocation)
    if(isGoalState(currentLocation)):
        return []
    availableSuccessors = getSuccessors(currentLocation)
    # traversing in depth
    for successor in availableSuccessors:
        if(successor[0] not in visited):
            if(depthFirstTraverse(successor)):
                break

    return directionSequence.list


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    isGoalState = problem.isGoalState
    # fn: (next state/location, action, cost)
    getSuccessors = problem.getSuccessors
    currentLocation = problem.getStartState()  # (x, y)

    # our data structure needed for dfs
    visited = []  # visited location list
    fringe = util.Queue()  # keeps actions from shallowest to deepest
    pathSequence = util.Queue()  # will register all path directions

    # initializing root node
    visited.append(currentLocation)
    goalStateCheck = isGoalState(currentLocation)
    if(goalStateCheck):
        return []
    pathSequence.push([])

    # traversing in breadth
    while(True):
        # expand created node in fringe
        if(not fringe.isEmpty()):  # first loop has empty fringe(no action at first)
            successor = fringe.pop()
            currentLocation = successor[0]
        # continuing last path
        currentPath = pathSequence.pop()

        # starting to create child nodes by expansion
        if(isGoalState(currentLocation)):  # check state before expansion
            return currentPath
        availableSuccessors = getSuccessors(currentLocation)

        for successor in availableSuccessors:
            if(successor[0] not in visited):
                fringe.push(successor)
                # visit happen during creation
                visited.append(successor[0])
                pathSequence.push(currentPath+[successor[1]])

        if(fringe.isEmpty()):  # target not found
            break

    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    isGoalState = problem.isGoalState
    # fn: (next state/location, action, cost)
    getSuccessors = problem.getSuccessors
    currentLocation = problem.getStartState()  # (x, y)

    # our data structure needed for dfs
    visited = []  # visited location list
    fringe = util.PriorityQueue()  # keeps actions from shallowest to deepest
    pathSequence = util.PriorityQueue()  # will register all path directions
    pathCost = {}  # will keep path cost update

    # initializing root node
    visited.append(currentLocation)
    goalStateCheck = isGoalState(currentLocation)
    if(goalStateCheck):
        return []
    pathSequence.push([], 0)
    pathCost[str(currentLocation)] = 0

    while(True):

        # expand created node in fringe
        if(not fringe.isEmpty()):  # first loop has empty fringe(no action at first)
            successor = fringe.pop()
            currentLocation = successor[0]
        # continuing last path
        currentPath = pathSequence.pop()
        cost = pathCost[str(currentLocation)]

        # starting to create child nodes by expansion
        if(isGoalState(currentLocation)):  # check state before expansion
            return currentPath
        availableSuccessors = getSuccessors(currentLocation)

        for successor in availableSuccessors:
            if((successor[0] not in visited) or
               # or if a lower cost exist to a visited node
               (pathCost.get(str(successor[0])) != None and
                    (cost+successor[2] < pathCost[str(successor[0])]))):
                pathCost[str(successor[0])] = cost+successor[2]
                fringe.push(successor, cost+successor[2])
                pathSequence.push(
                    currentPath+[successor[1]], cost+successor[2])
                # visit happen during creation
                visited.append(successor[0])

        if(fringe.isEmpty()):  # target not found
            break

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    isGoalState = problem.isGoalState
    # fn: (next state/location, action, cost)
    getSuccessors = problem.getSuccessors
    currentLocation = problem.getStartState()  # (x, y)
    def getHueristic(state): return heuristic(state, problem)

    # our data structure needed for dfs
    visited = []  # visited location list
    fringe = util.PriorityQueue()  # keeps actions from shallowest to deepest
    pathSequence = util.PriorityQueue()  # will register all path directions
    pathCost = {}  # will keep path cost update

    # initializing root node
    visited.append(currentLocation)
    goalStateCheck = isGoalState(currentLocation)
    if(goalStateCheck):
        return []
    pathSequence.push([], 0 + + getHueristic(currentLocation))
    pathCost[str(currentLocation)] = 0 + getHueristic(currentLocation)

    while(True):

        # expand created node in fringe
        if(not fringe.isEmpty()):  # first loop has empty fringe(no action at first)
            successor = fringe.pop()
            currentLocation = successor[0]
        # continuing last path
        currentPath = pathSequence.pop()
        cost = pathCost[str(currentLocation)]

        # starting to create child nodes by expansion
        if(isGoalState(currentLocation)):  # check/test state before expansion
            return currentPath
        availableSuccessors = getSuccessors(currentLocation)

        for successor in availableSuccessors:
            if((successor[0] not in visited) or
               # or if a lower cost exist to a visited node
               (pathCost.get(str(successor[0])) != None and
                    (cost + successor[2] < pathCost[str(successor[0])]))):
                newCost = cost + successor[2]  # g funtion 
                predictionCost = newCost + getHueristic(successor[0]) # f function
                pathCost[str(successor[0])] = newCost
                fringe.push(successor, predictionCost)
                pathSequence.push(
                    currentPath+[successor[1]], predictionCost)
                # visit happen during creation
                visited.append(successor[0])

        if(fringe.isEmpty()):  # target not found
            break

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
