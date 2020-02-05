# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
from heuristics import *
import random

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        DFS_stack = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using admissibleHeuristic heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in DFS_stack]
        # get best choice
        bestScore = min(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        
        legal = state.getLegalPacmanActions()
        # setting the flag in case None is returned
        boolean = 0                  
        BFS_queue = []
        for action in legal:
            BFS_queue.append((state.generatePacmanSuccessor(action),action))
        #using while loop with condition on the length of Queue (until length of queue > 0)
        while(len(BFS_queue))!=0:
            if (boolean == 1):
                break                      
            (route, action)=BFS_queue.pop(0)
            # if reached the win state
            if (route.isWin())!=0:
                return action
            #getting actions that pacman is allowed to take 
            legal = route.getLegalPacmanActions()
            # generating successors for following actions in legal actions
            for following_Action in legal:
                following_Succesor=route.generatePacmanSuccessor(following_Action)
                if (following_Succesor==None):
                    boolean=1
                    break
                    #adding following successor and action in the BFS Queue
                BFS_queue.append((following_Succesor,action))

        
        # When goal state is not attained, return action corresponding to the node with minimum heuristic value to goal
        
        if(len(BFS_queue))!=0:
            scored = [(admissibleHeuristic(state), action) for state, action in BFS_queue]
            # taking minimum value among the heuristic values
            bestScore = min(scored)[0]
            for pair in scored:
                if pair[0] == bestScore:
                    bestActions = pair[1]
                    break
            return bestActions
        return Directions.STOP





class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
    	legal = state.getLegalPacmanActions() 
        # setting the flag in case None is returned
        boolean = 0    
        DFS_stack = []               
        for action in legal:
            DFS_stack.append((state.generatePacmanSuccessor(action), action))
            #using while loop with condition on the length of Stack (until length of stack > 0)
        while(len(DFS_stack))!=0:
            if (boolean==1):
                break                      
            (path,action)=DFS_stack.pop(-1)
            # if reached the win state
            if (path.isWin())!=0:
                return action
            #getting actions that pacman is allowed to take 
            legal = path.getLegalPacmanActions()
            # generating successors for following actions in legal actions
            for following_action in legal:
                following_succesor = path.generatePacmanSuccessor(following_action)
                if (following_succesor==None):
                    boolean = 1
                    break
                    #adding following successor and action in the DFS Stack
                DFS_stack.append((following_succesor,action))
        # When goal state is not attained, return action corresponding to the node with minimum heuristic value to goal
        if (len(DFS_stack))!=0:
            scored = [(admissibleHeuristic(state), action) for state, action in DFS_stack]
            # taking minimum value among the heuristic values
            bestScore = max(scored)[0]
            for pair in scored:
                if pair[0]==bestScore:
                    bestActions=pair[1]
                    break
            return bestActions
        return Directions.STOP


class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        AStar_priority_queue = []
        depth=1
        legal= state.getLegalPacmanActions()
        # setting the flag in case None is returned
        boolean=0                  

        for action in legal:
            total_cost = depth - (admissibleHeuristic(state.generatePacmanSuccessor(action)) - admissibleHeuristic(state))
            AStar_priority_queue.append((total_cost,state.generatePacmanSuccessor(action) , action, depth))
            #using while loop with condition on the length of the priority queue (until length of priority queue > 0)
        while(len(AStar_priority_queue))!=0:
            if (boolean==1):
                break               
            (total_cost, path, action, depth)=AStar_priority_queue.pop(AStar_priority_queue.index(min(AStar_priority_queue)))
            # if reached the win state
            if (path.isWin())!=0:
                return action
            #getting actions that pacman is allowed to take 
            legal=path.getLegalPacmanActions()
            # generating successors for following actions in legal actions
            for following_action in legal:
                following_succesor = path.generatePacmanSuccessor(following_action)
                if (following_succesor == None):
                    boolean=1
                    break
                total_cost = (depth+1)-(admissibleHeuristic(path) - admissibleHeuristic(state))
                AStar_priority_queue.append((total_cost, following_succesor, action, depth + 1))

       # When goal state is not attained, return action corresponding to the node with minimum heuristic value to goal
        if(len(AStar_priority_queue))!=0:
            scored = [(admissibleHeuristic(state), action) for total_cost, state, action, depth in AStar_priority_queue]
             # taking minimum value among the heuristic values
            bestScore = min(scored)[0]
            for pair in scored:
                if pair[0] == bestScore:
                    bestActions=pair[1]
                    break
            return bestActions
        return Directions.STOP