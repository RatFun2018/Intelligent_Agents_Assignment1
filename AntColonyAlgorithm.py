import Data_Synthesizer as DS 
import numpy as np
import pandas as pd
import random as rd
import math
import matplotlib.pyplot as plt 

class Ant():
    def __init__(self,Employees,Tasks):
        self.Employees = Employees
        self.Tasks = Tasks
        self.solution_matrix = [[0*len(self.Employees)] for _ in range(len(self.Tasks))]

        for T in self.solution_matrix:
            c = rd.randint(0,len(T)-1)
            T[c] = 1

    def fitness(self):
        newcost = 0
    
        for E in self.Employees_Assigned:
            cumualitive_tasktime = 0
            not_skill = 0
            skilldiff =0
            over_Deadline = 0
            for T in E['Assigned Tasks']:
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


class AntColonyOptimser():
    def __init__(self,n_ants,a,evaporation,pheromone):
        self.ants = []
        self.a = a 
        self.evapaporation = evaporation 
        self.pheromones = pheromone
        self.pheromone_array = []
        for n in n_ants:
            newant = Ant()
            self.ants.append(newant)
        
        self.sol_shape = self.ants[0].solution_matrix.shape

        self.pheromone_array = [[1 * self.sol_shape[1]] for _ in range(self.sol_shape[0])]
    
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
        ...       
