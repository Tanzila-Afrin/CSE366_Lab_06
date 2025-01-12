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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        # Calculate the distance to the closest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        closestFoodDistance = min(foodDistances) if foodDistances else 0

        # Calculate the distance to the closest ghost
        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        closestGhostDistance = min(ghostDistances) if ghostDistances else 0

        # Calculate the score considering food and ghost distances
        score = successorGameState.getScore()
        score -= closestFoodDistance  # Prefer states with closer food
        if closestGhostDistance > 0:
            score += max(closestGhostDistance, 5)  # Prefer states with farther ghosts

        return score

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
        """
        "*** YOUR CODE HERE ***"
        action, score = self.minimax(0, 0, gameState)  # Get the action and score for pacman (agent_index=0)
        return action  # Return the action to be done as per minimax algorithm
    def minimax(self, curr_depth, agent_index, gameState):
        '''
        Returns the best score for an agent using the minimax algorithm. For max player (agent_index=0), the best
        score is the maximum score among its successor states and for the min player (agent_index!=0), the best
        score is the minimum score among its successor states. Recursion ends if there are no successor states
        available or curr_depth equals the max depth to be searched until.
        :param curr_depth: the current depth of the tree (int)
        :param agent_index: index of the current agent (int)
        :param gameState: the current state of the game (GameState)
        :return: action, score
        '''
        tmp = curr_depth
        indentation = "  " * curr_depth
        print(f"{indentation}Inside minimax------ curr_depth: {curr_depth} agent_index: {agent_index} ")
        # Roll over agent index and increase current depth if all agents have finished playing their turn in a move
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1
        # Return the value of evaluationFunction if max depth is reached
        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        # Initialize best_score and best_action with None
        best_score, best_action = None, None
        if agent_index == 0:  # If it is max player's (pacman) turn
            for action in gameState.getLegalActions(agent_index):  # For each legal action of pacman
                # Get the minimax score of successor
                # Increase agent_index by 1 as it will be next player's (ghost) turn now
                # Pass the new game state generated by pacman's `action`
                next_game_state = gameState.generateSuccessor(agent_index, action)
                
                _, score = self.minimax(curr_depth, agent_index + 1, next_game_state)
                
                # Update the best score and action, if best score is None (not updated yet) or if current score is
                # better than the best score found so far
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
                print(f"{indentation}curr_depth: {curr_depth} agent_index: {agent_index} action: {action}  next_game_state: { next_game_state} score:{score} best_score: {best_score}")
        else:  # If it is min player's (ghost) turn
            for action in gameState.getLegalActions(agent_index):  # For each legal action of ghost agent
                # Get the minimax score of successor
                # Increase agent_index by 1 as it will be next player's (ghost or pacman) turn now
                # Pass the new game state generated by ghost's `action`
                next_game_state = gameState.generateSuccessor(agent_index, action)
                
                _, score = self.minimax(curr_depth, agent_index + 1, next_game_state)
                print(f"{indentation}curr_depth: {curr_depth} agent_index: {agent_index} action: {action} next_game_state: { next_game_state} score:{score} best_score: {best_score}")
                # Update the best score and action, if best score is None (not updated yet) or if current score is
                # better than the best score found so far
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action
        # If it is a leaf state with no successor states, return the value of evaluationFunction
        
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        print(f"{indentation}Exit minimax------ curr_depth: {tmp} agent_index: {agent_index} best_action: {best_action} best_score: {best_score}")
        return best_action, best_score  # Return the best_action and best_score
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(curr_depth, agent_index, gameState, alpha, beta):
            if agent_index >= gameState.getNumAgents():
                agent_index = 0
                curr_depth += 1
            if curr_depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent_index == 0:
                value = float('-inf')
                for action in gameState.getLegalActions(agent_index):
                    value = max(value, alphaBeta(curr_depth, agent_index + 1, gameState.generateSuccessor(agent_index, action), alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                value = float('inf')
                for action in gameState.getLegalActions(agent_index):
                    value = min(value, alphaBeta(curr_depth, agent_index + 1, gameState.generateSuccessor(agent_index, action), alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        best_action = None
        alpha = float('-inf')
        beta = float('inf')
        value = float('-inf')
        for action in gameState.getLegalActions(0):
            new_value = alphaBeta(0, 1, gameState.generateSuccessor(0, action), alpha, beta)
            if new_value > value:
                value = new_value
                best_action = action
            alpha = max(alpha, value)
        return best_action

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
        def expectimax(curr_depth, agent_index, gameState):
            if agent_index >= gameState.getNumAgents():
                agent_index = 0
                curr_depth += 1
            if curr_depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent_index == 0:
                value = float('-inf')
                for action in gameState.getLegalActions(agent_index):
                    value = max(value, expectimax(curr_depth, agent_index + 1, gameState.generateSuccessor(agent_index, action)))
                return value
            else:
                value = 0
                legalActions = gameState.getLegalActions(agent_index)
                prob = 1 / len(legalActions)
                for action in legalActions:
                    value += prob * expectimax(curr_depth, agent_index + 1, gameState.generateSuccessor(agent_index, action))
                return value

        best_action = None
        value = float('-inf')
        for action in gameState.getLegalActions(0):
            new_value = expectimax(0, 1, gameState.generateSuccessor(0, action))
            if new_value > value:
                value = new_value
                best_action = action
        return best_action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This evaluation function considers the current score, the distance to the closest food,
      the distance to the closest ghost, and the number of remaining food pellets.
    """
    # Useful information you can extract from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Calculate the distance to the closest food
    foodDistances = [manhattanDistance(pos, foodPos) for foodPos in food.asList()]
    closestFoodDistance = min(foodDistances) if foodDistances else 0

    # Calculate the distance to the closest ghost
    ghostDistances = [manhattanDistance(pos, ghostState.getPosition()) for ghostState in ghostStates]
    closestGhostDistance = min(ghostDistances) if ghostDistances else 0

    # Calculate the score considering food and ghost distances
    score = currentGameState.getScore()
    score -= closestFoodDistance  # Prefer states with closer food
    if closestGhostDistance > 0:
        score += max(closestGhostDistance, 5)  # Prefer states with farther ghosts

    return score

# Abbreviation
better = betterEvaluationFunction

