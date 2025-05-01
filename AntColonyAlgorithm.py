import Data_Synthesizer as DS 
import numpy as np
import pandas as pd
import random as rd
import math
import matplotlib.pyplot as plt 
import copy

class Ant():
    def __init__(self,Employees,Tasks):
        self.Employees = Employees
        self.Tasks = Tasks
        self.solution_matrix = [[0 for _ in range(len(self.Employees))] for _ in range(len(self.Tasks))]
        self.cost = float("inf")
        for T in self.solution_matrix:
            c = rd.randint(0,len(T)-1)
            T[c] = 1
        print(self.solution_matrix)
        self._translate_solution()

    def _translate_solution(self):
        self.Employees_Assigned = copy.deepcopy(self.Employees)
        for f in self.Employees_Assigned:
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
        newcost = 0
    
        for E in self.Employees_Assigned:
            cumualitive_tasktime = 0
            not_skill = 0
            skilldiff =0
            over_Deadline = 0
            print(E)
            for T in E['Assigned Tasks']:
                print(f'Tasks:{T}')
                cumualitive_tasktime += E['Assigned Tasks'][T]['Estimated Time']
                if E['Assigned Tasks'][T]['Skills'] not in E['Skills']:
                    not_skill += 10

                if E['Assigned Tasks'][T]['Difficulty'] > E['Skill_lvl']:
                    skilldiff += 10

                over_Deadline = max(cumualitive_tasktime-E['Assigned Tasks'][T]['Deadline'],0)
            print(f'cumulative Task Time: {cumualitive_tasktime}')
            overtime = max(cumualitive_tasktime-E['Hours'],0)
            newcost += (0.25 * not_skill + 0.25 * skilldiff + 0.25 * over_Deadline + 0.25 * overtime)
        if newcost < self.cost:
            self.pBest = self.solution_matrix
        self.cost = newcost

    def update(self,pheremone_probability):
        selection_rd = rd.random()
        for i in range(len(pheremone_probability)):
            cumulative_prob = 0 
            j=0 
            while cumulative_prob < selection_rd:
                cumulative_prob += pheremone_probability[i][j]
                j += 1 
            self.solution_matrix[i] = [0 * len(self.Employees)]
            self.solution_matrix[i][j-1] = 1 
    
    def output(self):
        for E in self.Employees_Assigned:
            print(E)
        print(f'Cost: {self.cost}')
                

class AntColonyOptimser():
    def __init__(self,n_ants,a,evaporation,pheromone,patience=5):
        self.patience = patience
        self.patience_count = 0
        
        self.a = a 
        
        self.evapaporation = evaporation 
        self.pheromones = pheromone
        self.ants = []
        self.cost_history = []
        self.pheromone_array = []
        self.BestCost = float('inf')
        self.BestSolution = None
        _employees, _tasks  = DS.Generate_data(['A','B','C','D','E'],10,15)
        for n in range(n_ants):
            newant = Ant(_employees,_tasks)
            newant.output()
            self.ants.append(newant)
        
        self.sol_shape = [len(self.ants[0].solution_matrix),len(self.ants[0].solution_matrix[0])]

        self.pheromone_array = [[1 * self.sol_shape[0]] for _ in range(self.sol_shape[1])]

        while self.patience_count < self.patience:
            self.next()
        self.BestSolution.output()

    def calc_probability(self):
        self.proability_array = []
        for n in self.pheromone_array: 
            k_tau = sum(n)
            L2 = [x/k_tau for x in n]
            self.proability_array.append(L2)
            
    def evaporate(self):
        for p1 in self.pheromone_array:
            for p2 in p1: 
                p2 = (1-self.evapaporation)*p2
    
    def update_pheremone(self): 
        for p1 in range(len(self.pheromone_array)):
            for p2 in range(len(self.pheromone_array[p1])): 
                for ants in self.ants: 
                    self.pheromone_array[p1][p2] += ants.solution_matrix[p1][p2]*(self.evapaporation/ants.cost)

    def next(self):
        self.calc_probability()
        for A in self.ants: 
            A.update(self.pheromone_array)
            self.update_pheremone()
            self.evaporate()
            if A.cost <= self.BestCost:
                self.BestCost = A.cost
                self.BestSolution = A 
                self.cost_history.append(self.BestCost)
            else:
                self.patience_count += 1

    def plot_cost(self):
        plt.plot(self.cost_history,'b-',linewidth=3,label ='Best Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()




A = AntColonyOptimser(3,1,0.8,0.02)
A.plot_cost()