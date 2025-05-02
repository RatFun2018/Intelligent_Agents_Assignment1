import Data_Synthesizer as DS 
import numpy as np
import pandas as pd
import random as rd
import math
import matplotlib.pyplot as plt 
import psutil
import time
import copy
import os 

'''
Ant Colony Optimiser implementation using object oriented coding for job assigment problem 
this file contains the Ant() class object and the AntColonyOptimser() class object 

AntColonyOptimiser contains the process loop for optmising the problem and generates the desired of ants to find an optimal solution to the problem


'''

class Ant():
    '''
    Ant class object that is used in AntColony Optimiser contains functions to update solution
    Ant is intiated with a randomised solution 
    '''
    def __init__(self,Employees,Tasks):
        self.Employees = Employees
        self.Tasks = Tasks
        self.solution_matrix = [[0 for _ in range(len(self.Employees))] for _ in range(len(self.Tasks))] #Intialising solution_matrix into the correct shape 

        #Intialisng metric tracking variables 
        self.cost = float("inf")
        self.skill_lvl_violation = 0 
        self.skill_violation = 0 
        self.overtime_violation = 0 
        self.deadline_violation = 0 
        
        for T in self.solution_matrix: #Randomising Intial solution on creation of ant object this method ensures that each task is only assigned to one employee 
            c = rd.randint(0,len(T)-1)
            T[c] = 1
        #print(self.solution_matrix)
        self._translate_solution()

    def _translate_solution(self):
        '''
        Function that translates the solution binary matrix into a Dictionary Format of displaying the task assignmnet information 
        Implementation of the cost calculation was easier with a dictionary data format but other functions such as pheremone calculation 
        and path determination was better done in a matrix format 
        '''
        self.Employees_Assigned = copy.deepcopy(self.Employees)
        for f in self.Employees_Assigned: #Resets the task assignments in the Employee_Assigned 
            f['Assigned Tasks'] = {}
        task_idx = 0

        for T in self.solution_matrix:
            employee_idx = 0
            for E in T:
                if E == 1:
                    #print(f'task_idx: {task_idx}')
                    taskname = 'T' + str(task_idx)
                    self.Employees_Assigned[employee_idx]['Assigned Tasks'].update({taskname:self.Tasks[task_idx]})
            #print(f'Task{task_idx} Assigned to Employee{employee_idx}')
                employee_idx +=1
            task_idx +=1
        self.fitness()

    def fitness(self):
        '''
        Determines the Cost function for the data after being traslated from the binary Matrix
        '''
        newcost = 0

        # Metric Tracking Variables reset every call of fitness() as to not increase over iterations of optimiser 
        self.skill_lvl_violation = 0 
        self.skill_violation = 0 
        self.overtime_violation = 0 
        self.deadline_violation = 0 


        for E in self.Employees_Assigned:
            # Cost metrics reset for every employee 
            cumualitive_tasktime = 0
            not_skill = 0
            skilldiff =0
            over_Deadline = 0
            for T in E['Assigned Tasks']:
                #print(f'Tasks:{T}')
                cumualitive_tasktime += E['Assigned Tasks'][T]['Estimated Time']
                if E['Assigned Tasks'][T]['Skills'] not in E['Skills']:
                    not_skill += 1
                    self.skill_violation += 1 

                if E['Assigned Tasks'][T]['Difficulty'] > E['Skill_lvl']:
                    skilldiff += 1
                    self.skill_lvl_violation += 1 

                over_Deadline = max(cumualitive_tasktime-E['Assigned Tasks'][T]['Deadline'],0)
                if over_Deadline != 0: 
                    self.deadline_violation += 1 
            #print(f'cumulative Task Time: {cumualitive_tasktime}')
            overtime = max(cumualitive_tasktime-E['Hours'],0)
            if overtime != 0: 
                self.overtime_violation += 1 
            newcost += (0.25 * not_skill + 0.25 * skilldiff + 0.25 * over_Deadline + 0.25 * overtime) # Adds the cost that the tasks assigned to the employee to a running counter cost for all employees in this iteration 
        
        self.cost = newcost 

    def update(self,pheremone_probability):
        '''
        updating path assignment using the pheremone_probability matrix from an optimiser. Uses roulette selection to randomly sele
        '''
        for i in range(len(pheremone_probability)):
            selection_rd = rd.random()
            self.solution_matrix[i] = [0 for _ in  range(len(self.Employees))]
            probs = pheremone_probability[i]
            cumulative_probs = 0.0 
            for j,p in enumerate(probs):
                cumulative_probs += p
                if selection_rd <= cumulative_probs:
                    self.solution_matrix[i][j] = 1
                    break  
        
        self._translate_solution()
        
         
    
    def output(self):
        '''
        Simple output function that prints the current state information of the ant object 
        '''
        idx = 0
        for E in self.Employees_Assigned:
            print(f'Employee{idx}:{E}')
            idx +=1 
        print(f'Cost: {self.cost}')
        print(f'Solution Matrix: {self.solution_matrix}')
                

class AntColonyOptimser():
    '''
    AntColonyOptimise Class
    Holds all functions and variables required to do a Antcolony optmiser for the job assignment problem
    Inputs:
    n_ants: Number of ants to be generated and used in the optmiser 
    evaporation: rate of evaporation of pheremone trails expects a float value that is <1 
    pheremone: value that effects the rate that pheremones are applied to chosen paths expected values between 


    format for employee and task information is outlied in the data systhesizer method 
    employees: list type object containing employee information 
    tasks: list type object containing task information 

    a: value used in the probability calculation for simplcity set as 1 
    patience: number of iterations without improvements to wait for before stopping 

    '''
    def __init__(self,n_ants,evaporation,pheromone,employees,tasks,n_iter=50,a=1 ):
        #self.patience = patience
        self.patience_count = 0
        
        self.a = float(a) 
        
        self.evapaporation = evaporation #
        self.pheromones = pheromone
        self.ants = [] #List that holds all the ant objects 
        
        self.pheromone_array = []


        # Variables for Recording performance metrics through the iteration of the algoritm 
        self.cost_history = [] 
        self.averagetotalViolatioHist = []
        self.skill_violationHist = []
        self.skill_lvl_violationHist = [] 
        self.deadline_violationHist = []
        self.overtime_violationHist = []
        self.BestCost = float('inf') #Best cost initially inifinite 
        self.BestSolution = None
        # Memory and time metric logging 
        self.memoryuseHist = []
        self.process_timeHist =[]
        self.process = psutil.Process(os.getpid())

        
        for n in range(n_ants): #Intialising the ants each containing a unique solution 
            newant = Ant(employees,tasks)
            #newant.output()
            self.ants.append(newant)
        
        # Intialising 
        self.sol_shape = [len(self.ants[0].solution_matrix),len(self.ants[0].solution_matrix[0])]
        #print(f'Solution Shape {self.sol_shape}')
        self.pheromone_array = [[float(1) for _ in range(self.sol_shape[1])] for _ in range(self.sol_shape[0])]
        #print(f'Phaeremone array: {self.pheromone_array}')

        #while self.patience_count < self.patience: #Uses patience matric to continue optmising until (patience) amount of iterations without improvement on the  best cost 
            #self.next()
            #print('='*50)

        for n in range(n_iter):
            self.next()
            print('='*50)
        print("Best Solution:")
        self.BestSolution.output()

    def calc_probability(self): #Calculating proability matrix from pheremone matrix 

        self.proability_array = []
        for n in self.pheromone_array: 
            k_tau = sum(n)
            L2 = [(np.float_power(x,self.a))/k_tau for x in n] #probability function 
            self.proability_array.append(L2)
            
    def evaporate(self): #Applies evaporation effect to all pheremone values 
        for p1 in self.pheromone_array:
            for p2 in p1: 
                p2 = (1-self.evapaporation)*p2 #pheremones are equally reduced at every itteration by evaporation 
    
    def update_pheremone(self): 
        for p1 in range(len(self.pheromone_array)):
            for p2 in range(len(self.pheromone_array[p1])): 
                for ants in self.ants: 
                    #print(ants.solution_matrix)
                    self.pheromone_array[p1][p2] += ants.solution_matrix[p1][p2]*self.pheromones/ants.cost #updating pheremone array based on the formula taking into account pheremone value and path cost

    def next(self):
        self.calc_probability()

        #Resets variable for logging metric per iteration 
        newBestCost = float('inf')
        start_time = time.time()    
        avg_total_violation = 0 
        avg_skill_violation = 0 
        avg_skill_lvl_violation = 0 
        avg_deadline_violation = 0 
        avg_overtime_violation = 0
        for A in self.ants:
            #print(f'Ant Solution Matrix Before: {A.solution_matrix}')
            #print(f'Probability Array: {self.proability_array}') 
            A.update(self.proability_array)
            
            #logging of violation values of all Ants in the iteration
            avg_total_violation += sum([A.skill_lvl_violation,A.skill_violation,A.deadline_violation,A.overtime_violation])
            avg_skill_lvl_violation += A.skill_lvl_violation
            avg_skill_violation += A.skill_violation 
            avg_deadline_violation += A.deadline_violation 
            avg_overtime_violation += A.overtime_violation 

            if A.cost <= newBestCost:
                newBestCost = A.cost
                newBestSolution = A 
            #print(f'Ant Solution Matrix After: {A.solution_matrix}')
        self.update_pheremone()
        #print(f'After Pheremone update {self.pheromone_array}\n')
        self.evaporate()
        #print(f'After evaporation: {self.pheromone_array}\n')

        # find the man violation in all the ants for the iteration 
        avg_total_violation = avg_total_violation/len(self.ants) 
        avg_skill_lvl_violation = avg_skill_lvl_violation/len(self.ants)
        avg_skill_violation = avg_skill_violation/len(self.ants) 
        avg_deadline_violation = avg_deadline_violation/len(self.ants) 
        avg_overtime_violation = avg_overtime_violation/len(self.ants) 

        # Logging violation values so they can be graphed 
        self.averagetotalViolatioHist.append(avg_total_violation) 
        self.skill_lvl_violationHist.append(avg_skill_lvl_violation)
        self.skill_violationHist.append(avg_skill_violation) 
        self.deadline_violationHist.append(avg_deadline_violation)
        self.overtime_violationHist.append(avg_overtime_violation)
        
        if newBestCost < self.BestCost: 
            self.BestCost = newBestCost
            self.BestSolution = newBestSolution
            self.patience_count = 0 
        else :  
            self.patience_count +=1 
            print("No improvement")
            print(f"Patience {self.patience_count}/{self.patience}")
            self.cost_history.append(self.BestCost)
        
        end_time = time.time()
        end_mem =  self.process.memory_info().rss / 1024
        iteration_time  = end_time - start_time # comparing time taken from start of update to end of update to determine algorithm process time 
        self.memoryuseHist.append(end_mem)
        self.process_timeHist.append(iteration_time)


    def plot_cost(self):
        '''
        Simple quick and dirty graphing function that tries to display or logged metrics over the number of iterations of the optimiser 
        '''

        plt.subplot(3,2,1)
        plt.plot(self.cost_history,'b-',linewidth=3,label ='Best Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.subplot(3,2,2)
        plt.plot(self.averagetotalViolatioHist,'r-',linewidth=3,label='total avg Violations')
        plt.xlabel('Iterations')
        plt.ylabel("# of violations")
        plt.subplot(3,2,3)
        plt.plot(self.skill_lvl_violationHist,'o-',linewidth= 2,label='skill lvl violations')
        plt.plot(self.skill_violationHist,'g-',linewidth = 2, label= 'Skill Violations')
        plt.plot(self.deadline_violationHist,'r-',linewidth = 2 , label='Deadline Violation')
        plt.plot(self.overtime_violationHist,'b-',linewidth = 2 , label = 'Overtime Violation')
        plt.xlabel('Iterations')
        plt.ylabel('# of violations')
        plt.subplot(3,2,4) 
        plt.plot(self.memoryuseHist,'g-')
        plt.xlabel('Iterations')
        plt.ylabel('Memory Used kb',color='tab:green')
        plt.subplot(3,2,5)
        plt.plot(self.process_timeHist,'r-')
        plt.xlabel('Iterations')
        plt.ylabel('Time Used',color='tab:red')
        plt.show()


# Ant_employees, Ant_Tasks = DS.Generate_data(['A','B','C','D','E'],10,25)
# A = AntColonyOptimser(5,1,0.8,0.02,Ant_employees,Ant_Tasks,patience=100)

# A.plot_cost()