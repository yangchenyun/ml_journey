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


import math
from statistics import mean

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent, Directions
from pacman import GameState
from search import mazeDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        scores = [int(s * 1e5) for s in scores]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        # "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        walls = currentGameState.getWalls()
        top, right = walls.height - 2, walls.width - 2
        corners = ((1, 1), (1, top), (right, 1), (right, top))

        def wall_test(pos):
            x, y = pos
            result = walls[x][y]

        food_list = newFood.asList()
        ghost_list = [
            (
                int(state.configuration.getPosition()[0]),
                int(state.configuration.getPosition()[1]),
            )
            for state in newGhostStates
        ]

        # IDEA 1 - The evaluation is a linear combination of the following features:
        #
        # - Manhattan distance to remaining food (TODO: use MST)
        # - Distance to ghost (A* search), if scare time
        # - Distance to capsule (A* search)
        # - Scores
        # - ...
        #
        total_score = 0.0

        score_w = 10.0
        score = currentGameState.getScore() * score_w
        total_score += score

        # Worked VERY WELL!!
        ghost_w = lambda t: 0.0 if t > 0 else -10.0
        ghost_dis_fn = lambda dis: 1 / (dis - 0.5) ** 2 if dis != 0 else 1e10
        ghost_dis_all = [
            mazeDistance(successorGameState, newPos, pos) for pos in ghost_list
        ]
        ghost_score = mean(
            ghost_w(t) * ghost_dis_fn(dis)
            for dis, t in zip(ghost_dis_all, newScaredTimes)
        )
        for t in newScaredTimes:
            if t <= 1:  # be careful when the timers is up
                total_score += ghost_score
        print("ghost_score:", ghost_score)

        if food_list:
            # Food policy, there is a bug
            food_w = -1 / 10.0
            # food_fn = lambda dis: min(
            #     1.0, 1 / dis**2 if dis != 0 else 1e10
            # )  # capped at 1.0, the closer the better
            food_fn = lambda dis: math.log(dis)
            # food_fn = lambda dis: -1 / (dis + 0.5) ** 2 if dis != 0 else 1e10
            food_dis = sorted(manhattanDistance(newPos, food) for food in food_list)
            min_food_dis = food_dis[0]
            # NOTE: if using mean, will have issues towards the end
            food_score = sum([food_w * food_fn(dis) for dis in food_dis])
            total_score += food_score

            # Food count to encourage eating
            food_c_w = -1.0
            food_c_score = food_c_w * len(food_list)
            total_score += food_c_score

            # Prioritize corner food
            corner_w = -1.0 / 3.0
            food_corners = [c for c in corners if c in food_list]
            corner_dis = [manhattanDistance(newPos, corner) for corner in food_corners]
            corner_score = sum([corner_w * food_fn(dis) for dis in corner_dis])
            total_score += corner_score
            print("corner_score:", corner_score)

            # Add noise to break ties when close
            if min_food_dis < 2:
                seed = {
                    Directions.NORTH: 1,
                    Directions.SOUTH: 2,
                    Directions.EAST: 3,
                    Directions.WEST: 4,
                    Directions.STOP: 5,
                }
                random.seed(
                    total_score + seed[action] * 100 + newPos[0] * 10 + newPos[1]
                )
                random_score = 1.0 * random.random()
                total_score += random_score

            print("food_score:", food_score)

        # penalize no action
        # stop_w = -1.0
        # stop_score = stop_w * (1.0 if action == Directions.STOP else 0.0)
        # total_score += stop_score

        if newCapsules:
            capsule_w = -1.0
            capsule_dis = [
                manhattanDistance(newPos, capsule) for capsule in newCapsules
            ]
            capsule_score = sum([capsule_w * food_fn(dis) for dis in capsule_dis])
            total_score += capsule_score
            print("capsule_score:", capsule_score)

        # print(f"total_score: {total_score}")
        return total_score

        #
        # IDEA 2 - Zone Boost
        #

        def zone_expand(pos, r=1):
            top, right = walls.height - 2, walls.width - 2
            positions = [
                (int(pos[0] + x), int(pos[1] + y))
                for y in range(-r, r + 1)
                for x in range(-r, r + 1)
            ]
            positions = [
                (x, y)
                for x, y in positions
                if (x >= 0 and x <= right)
                and (y >= 0 and y <= top)
                and (not walls[x][y])
            ]
            return positions

        dead_zone = [
            pos for g in ghost_list for pos in zone_expand(g, r=1)
        ] + ghost_list
        danger_zone = [pos for g in ghost_list for pos in zone_expand(g, r=2)]
        bonus_zone = [pos for g in newCapsules for pos in zone_expand(g, r=2)]
        food_zone = [pos for f in food_list for pos in zone_expand(f, r=5)]
        neighbors = [
            (newPos[0] + pos[0], newPos[1] + pos[1])
            for pos in [(0, 1), (1, 0), (0, -1), (-1, 0)]
        ]

        # Food score idea, to break the evenness and encourage action
        # It is computed everytime and still result in stillness
        #
        # food_scores = {}
        # for pos in food_zone:
        #     food_scores[pos] = food_scores.get(pos, 0)
        #     food_scores[pos] += random.random()

        # print(
        #     "newPos: ",
        #     newPos,
        #     "dead_zone: ",
        #     dead_zone,
        #     "danger_zone: ",
        #     danger_zone,
        #     "ghost:",
        #     ghost_list,
        #     "expanded:",
        #     zone_expand(ghost_list[0], r=1),
        # )
        #
        #
        # How to penalize no actions?
        # How to break ties when making a decision?

        # dead_score = -5 * dead_zone.count(newPos)
        # danger_score = -3 * danger_zone.count(newPos)
        # bonus_score = 5 * bonus_zone.count(newPos)
        # capsule_score = 20 * newCapsules.count(newPos)
        # food_score = 0.05 * food_zone.count(newPos)
        # # postion bias to break ties
        # bias_score = 0.01 * newPos[0] + 0.01 * newPos[1]
        # # penalize if not eat neighbor food
        # neighbor_food_penalty = random.random() * sum(
        #     [food_list.count(nei) for nei in neighbors]
        # )
        # stop_penalty = 5 if action == Directions.STOP else 0.0

        # print(
        #     "Scores: ",
        #     dead_score,
        #     danger_score,
        #     bonus_score,
        #     food_score,
        #     capsule_score,
        #     bias_score,
        #     -neighbor_food_penalty,
        #     -stop_penalty,
        # )

        # score = currentGameState.getScore()
        # score = (
        #     score
        #     + bonus_score
        #     + food_score
        #     + capsule_score
        #     + bias_score
        #     - neighbor_food_penalty
        #     # - food_dis_min
        #     - newFood.count()
        # )
        # # - food_dis_min

        # for t in newScaredTimes:
        #     if t > 1:  # be careful when the timers is up
        #         score += dead_score + danger_score

        # for t in newScaredTimes:
        #     if t > 0:
        #         score += 100  # compensate for capsure loss

        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
