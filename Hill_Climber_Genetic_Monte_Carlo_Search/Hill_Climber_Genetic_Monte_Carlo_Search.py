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
import math

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

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.rand_act = 50
        self.seq_act = 5
        self.actionList = []
        #Filling the list of actions from start
        for i in range(0,self.seq_act):
            self.actionList.append(Directions.STOP)                                      
        return 
    
    def getAction(self, state):
        mod_fwd = 1
        #assigning the score of the root to cal_score
        cal_score = scoreEvaluation(state)    
        #Assigning the list of all actions in temp_seq_act                                           
        temp_seq_act = self.actionList[:]                                          
        #getting the list of all the possible actions that Pacman can take
        self.action = state.getAllPossibleActions()                                     
        for i in range(0,len(self.actionList)):
            self.actionList[i] = self.action[random.randint(0,len(self.action)-1)]       
        #Running while loop until value of forward model doesn't get zero
        while(mod_fwd):
            #Assigning the current state to temporary state variable 
            tempState = state
            #Running for loop from 0 to the no. of actions in temp_seq_act
            for i in range(0, len(temp_seq_act)):
                #Generarting the list of succesors for each action in temp_seq_act array
                further_succesor = tempState.generatePacmanSuccessor(temp_seq_act[i]) 
                #watching if there is no succesor
                if(further_succesor == None):                                               
                    mod_fwd = 0
                    break
                #checking if the new succesor is in either Win or Lose State, if so, we terminate
                elif(further_succesor.isLose() or further_succesor.isWin()):                                          
                    break
                else:
                   tempState = further_succesor
            #if there is no succesor further then calculating the highest score
            if (mod_fwd == 0 and scoreEvaluation(tempState) >= cal_score):                                                     
                self.actionList = temp_seq_act[:]
                cal_score = scoreEvaluation(tempState)
                break
            #if there is succesor then calculating the highest score
            elif (mod_fwd == 1 and scoreEvaluation(tempState) >= cal_score):
                  cal_score = scoreEvaluation(tempState)
                  self.actionList = temp_seq_act[:]
            for i in range(0, len(temp_seq_act) > 0):
                #producing a random no. between one and hundred
                rand = random.randint(1,100)                                       
                if (rand >= self.rand_act):
                    temp_seq_act[i] = self.action[random.randint(0,len(self.action)-1)] 
                else:
                    temp_seq_act[i] = self.actionList[i]  
        #giving the action at 1st index back                        
        return self.actionList[0] 
 

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.seq_act = 5
        self.actionList = []
        self.len_pop = 8
        for i in range(self.seq_act):
            self.actionList.append(Directions.STOP)
        return
        self.cros_over = 70
        self.children_mut = 10
        self.cros_over_children = 50
    #generating the next population
    def generate_next_population(self, fit_chromosms):
        #choosing the parents
        choose_parent = [0,0]
        #creating an array for the new population
        pop_new = []   
        #Running for loop for half of the population     
        for j in range(0, self.len_pop/2):
           #for every p in 0 to 2
            for p in range(0, 2): 
                rnk_select = random.randint(1, 36)
                #conditions on rnk_select
                if rnk_select <= 1:
                    choose_parent[p] = fit_chromosms[7][1]
                elif rnk_select <= 3:
                    choose_parent[p] = fit_chromosms[6][1]
                elif rnk_select <= 6:
                    choose_parent[p] = fit_chromosms[5][1]
                elif rnk_select <= 10:
                    choose_parent[p] = fit_chromosms[4][1]
                elif rnk_select <= 15:
                    choose_parent[p] = fit_chromosms[3][1]
                elif rnk_select <= 21:
                    choose_parent[p] = fit_chromosms[2][1]
                elif rnk_select <= 28:
                    choose_parent[p] = fit_chromosms[1][1]
                elif rnk_select <= 36:
                    choose_parent[p] = fit_chromosms[0][1]            
            #Applying a random test when crosover is less than 70%
            if (random.randint(1,100) <= self.cros_over):
                #Taking array for the first child
                firstchild = []
                #Taking array for the second child
                secondchild = []
                #generating the first and second child
                for i in range(self.seq_act):
                    if random.randint(1, 100) <= self.cros_over_children:
                        firstchild.append(choose_parent[0][i])
                        secondchild.append(choose_parent[1][i])
                    elif random.randint(1, 100) >= self.cros_over_children:
                        firstchild.append(choose_parent[1][i])
                        secondchild.append(choose_parent[0][i])      
            elif(random.randint(1,100) >= self.cros_over):
                firstchild = choose_parent[0]
                secondchild = choose_parent[1]
            pop_new.append(firstchild)
            pop_new.append(secondchild)
            #Returning the next population
            return pop_new

    def getAction(self, state):                         
        mod_fwd = 1
        #Assigning the current state to temporary state variable 
        tempState = state
        #An array storing the population
        population = []
        #getting the list of all the possible actions that Pacman can take
        self.action = state.getAllPossibleActions()
        #for loop for each i in the size of poplation
        for i in range(0, self.len_pop):
            #for every j in 0 to size of the action list
            for j in range(0,len(self.actionList)):
                self.actionList[j] = self.action[random.randint(0,len(self.action)-1)];
            population.append(self.actionList)
        #Assigning population to precursory population
        self.population_precur = population[:][:]
        #An array for last state
        state_end = []
        for i in range(self.len_pop):
            state_end.append(state)
        self.prev_state_end = state_end[:]
        #Running while loop until value of forward model doesn't get zero
        while(mod_fwd):
            #Running the for loop for the size of population
            for i in range(0, self.len_pop):
                tempState = state
                #Running for loop for the action sequence
                for j in range(0, self.seq_act):
                    further_succesor = tempState.generatePacmanSuccessor(population[i][j])
                    #checking if there is no further succesor, then assigning temp state to end state
                    if(further_succesor == None):
                        mod_fwd = 0
                        break 
                    #checking whether pacman loses or wins
                    elif(further_succesor.isLose() or further_succesor.isWin()): 
                        break
                    else:
                       tempState = further_succesor
                state_end[i] = tempState
                if (mod_fwd == False):
                    break
            if (mod_fwd == False):
                break    
            #An array for population fitness
            pop_fit = []
            #calculating the score of each end state
            for i in range(0, self.len_pop):
                last_state = state_end[i]
                score_fit = scoreEvaluation(last_state)
                pop_fit.append((score_fit,population[i]))
            #Sorting the population fitness
            pop_fit.sort()
            #Reverse ordering
            pop_fit.reverse()
            rnking = 1
            #an array for fitness of chromosomes
            fit_chromosms = []
            for i in range(self.len_pop):
                fit_chromosms.append((rnking, pop_fit[i][1]))
                rnking=sum([1,rnking])
            pop_new = []
            for new_child in pop_new:
                if random.randint(1, 100) <= self.new_child_mut:
                    new_child[random.randint(0, len(new_child)-1)] = random.choice(self.action)
            self.population_precur = population[:][:]
            self.prev_state_end = state_end[:]
        #checking whether the new population is None is returned, if so, we take the old population
        if (mod_fwd == 0):
            population = self.population_precur[:][:]
            state_end = self.prev_state_end[:]
        pop_fit_end = []
        for i in range(self.len_pop):
            last_state = state_end[i]
            score_fit = scoreEvaluation(last_state)
            pop_fit_end.append(score_fit)
            score_max = max(pop_fit_end)
        
        sequnce = pop_fit_end.index(score_max)
        act_first = population[sequnce]
        #Returing the 1st action
        return act_first[0]

     
class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.const = 1
        self.roll_rand = 5
        return

    def getAction(self, state):    
        self.flag = False
        node = Node(state)
        b_n = node
        visit_most = 0
        self.mod_fwd = True 
        #Running while loop untill forward model = 1
        while(self.mod_fwd):
            node_new = self.treePolicy(node, state)
            #setting flag is false
            self.flag = False
            if (self.flag == 1):
                continue
            if (self.mod_fwd == False):
                break
            #Storing the result of default policy in the variable var
            var = self.defaultPolicy(node_new, state)
            if (self.mod_fwd == False):
                break
            if (self.flag == True):
                continue
            self.backProp(node_new, var)
        for i in range(0, len(node.chldrn) > 0):
            node_chldrn = node.chldrn[i]
            if (visit_most < node_chldrn.visit_cnt):
                b_n = node_chldrn
                visit_most = node_chldrn.visit_cnt            
        #Returning the action corresponding to best node
        return b_n.act

    #defining a method for selecting the best node
    def bestNode(self, node):
        #Assigning node to best node
        b_n = node
        #Initioalizing UCB variable
        UCB = 0
        for i in range(0, len(node.chldrn) > 0):
            node_chldrn = node.chldrn[i]
            if (node_chldrn.visit_cnt==0):
                node_chldrn.visit_cnt=1
            #Calculating the value of UCB using the formula
            val_UCB = sum([(node_chldrn.rwd/node_chldrn.visit_cnt), (self.const*math.sqrt(2*math.log(node.visit_cnt)/node_chldrn.visit_cnt))])
            #checking if the UCB value is less than recently calculated UCB, then update UCB
            if UCB < val_UCB :
                b_n = node_chldrn
                UCB = val_UCB
        #returning the best node        
        return b_n
    #defining a method for the default policy
    def defaultPolicy(self, node, state):
        node_total = []
        act_previous = []
        #running while loop until there is no parent
        while(node.node_parent):
            node = node.node_parent
            act_previous.append(node.act)
            node_total.append(node)
        #Assigning the current state to temporary state variable 
        tempState = state   
        #Running for loop for the previous actions
        for i in range(0, len(act_previous)):
            #poping out last node
            node_go_Back = node_total.pop(-1)
            #poping out last node
            act = act_previous.pop(-1)
            #Generating Pacman Succesor based on action
            chldrn = tempState.generatePacmanSuccessor(act)
            #checking if there is child or not
            if (chldrn == None):
                self.mod_fwd = False
                return
            elif(sum([chldrn.isWin(),chldrn.isLose()])==True):
                self.flag = True
                #Evaluating for reward
                rwd = gameEvaluation(state, chldrn)
                self.backProp(node_go_Back, rwd)
                return rwd
            else:
                tempState = chldrn
        #assigning node to a temporary node
        node_temporary = node
        for i in range(0, self.roll_rand):
            chldrn = tempState.generatePacmanSuccessor(random.choice(tempState.getLegalPacmanActions()))
            #checking if there is child or not
            if (chldrn == None):
                self.mod_fwd = False
                break
            elif(sum([chldrn.isWin(),chldrn.isLose()])==True):
                self.flag = True
                #Evaluating for reward
                rwd = gameEvaluation(state, chldrn)
                self.backProp(node_temporary, rwd)
                #Returning the reward
                return rwd
            else:
                tempState = chldrn
        rwd = gameEvaluation(state, tempState)
        return rwd
    #defining the backpropagation method       
    def backProp(self, node, rwd):
        #running unconditional while loop
        while 1: 
            node.rwd+=rwd
            node.visit_cnt = sum([node.visit_cnt,1])
            #checking if there is no parent node
            if (node.node_parent==None):
                break
            node = node.node_parent 
    #defining method corresponding to the tree policy
    def treePolicy(self, node, state):
        #Running while loop until forward model is not equal to zero
        while(self.mod_fwd):
            val = len(node.act_not_trvrsed)
            if (val):
                return self.nodeExpansion(node, state)
            else:
                #assigning node to the precursory node
                l_n = node
                node = self.bestNode(node)
                #checking if the node is same as the precursory node
                if node is l_n:
                    break
        return node               
    #defining method for the nodeExpansion of the node           
    def nodeExpansion(self, node, state):
        #Assigning node to a temporary node
        node_temporary = node
        act = random.choice(node_temporary.act_not_trvrsed)
        #Making an array contatining total nodes
        node_total = []
        #Assigning the current state to temporary state variable 
        tempState = state
        #Making an array contatining all past actions
        act_previous = []    
        #Running while loop until there is no parent node   
        while(node.node_parent):
            node_total.append(node)
            act_previous.append(node.act)
            node = node.node_parent
        #Running for loop for all the past actions
        for i in range(0, len(act_previous) > 0):
            #poping out last node
            node_go_Back = node_total.pop(-1)
            #poping out last action
            act_go_Back = act_previous.pop(-1)
            # generating pacman successor based on previous action
            chldrn = tempState.generatePacmanSuccessor(act_go_Back)
            #checking whether there is child or not
            if (chldrn == None):
                self.mod_fwd = False
                return
            elif(sum([chldrn.isWin(),chldrn.isLose()])==True):
                #Evaluating for reward
                rwd = gameEvaluation(state, chldrn)
                self.flag = True
                self.backProp(node_go_Back, rwd)
                return node_go_Back
            else:
                tempState=chldrn
        #Sureying child
        chldrn_survey = tempState.generatePacmanSuccessor(random.choice(node_temporary.act_not_trvrsed))
        if (chldrn_survey==None):
            self.mod_fwd = False
            return
        elif(sum([chldrn_survey.isWin(),chldrn_survey.isLose()]) == True):
            #Evaluating for reward
            rwd = gameEvaluation(state, chldrn_survey)
            self.flag = True
            self.backProp(node_temporary, rwd)
            #returing the temporary node
            return node_temporary
        else:
            node_temporary.act_not_trvrsed.pop(node_temporary.act_not_trvrsed.index(act))
            node_new = Node(chldrn_survey, node_temporary, act)
            node_temporary.chldrn.append(node_new)
            #returning the further node
            return node_new
    #defining the backpropagation method       
    def backProp(self, node, rwd):
        #running unconditional while loop
        while 1: 
            node.rwd+=rwd
            node.visit_cnt = sum([node.visit_cnt,1])
            #checking if there is no parent node
            if (node.node_parent==None):
                break
            node = node.node_parent 
#defining a class node
class Node():
    def __init__(self, state, node_parent = None, act = None):
        self.act_not_trvrsed = state.getLegalPacmanActions()
        self.act = act
        self.visit_cnt = 0
        self.chldrn = []
        self.rwd = 0
        self.node_parent = node_parent
