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
        self.skill_lvl_violation = 0 
        self.skill_violation = 0 
        self.overtime_violation = 0 
        self.deadline_violation = 0 
        for T in self.solution_matrix:
            c = rd.randint(0,len(T)-1)
            T[c] = 1
        #print(self.solution_matrix)
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
        self.skill_lvl_violation = 0 
        self.skill_violation = 0 
        self.overtime_violation = 0 
        self.deadline_violation = 0 
        for E in self.Employees_Assigned:
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
            newcost += (0.25 * not_skill + 0.25 * skilldiff + 0.25 * over_Deadline + 0.25 * overtime)
        if newcost < self.cost:
            self.pBest = self.solution_matrix
        self.cost = newcost

    def update(self,pheremone_probability):
        
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
        idx = 0
        for E in self.Employees_Assigned:
            print(f'Employee{idx}:{E}')
            idx +=1 
        print(f'Cost: {self.cost}')
        print(f'Solution Matrix: {self.solution_matrix}')
                

class AntColonyOptimser():
    def __init__(self,n_ants,a,evaporation,pheromone,employees,tasks,patience=10):
        self.patience = patience
        self.patience_count = 0
        
        self.a = float(a) 
        
        self.evapaporation = evaporation 
        self.pheromones = pheromone
        self.ants = []
        self.cost_history = []
        self.pheromone_array = []
        self.averagetotalViolatioHist = []
        self.skill_violationHist = []
        self.skill_lvl_violationHist = [] 
        self.deadline_violationHist = []
        self.overtime_violationHist = []
        self.BestCost = float('inf')
        self.BestSolution = None
        for n in range(n_ants):
            newant = Ant(employees,tasks)
            #newant.output()
            self.ants.append(newant)
        
        self.sol_shape = [len(self.ants[0].solution_matrix),len(self.ants[0].solution_matrix[0])]
        print(f'Solution Shape {self.sol_shape}')
        self.pheromone_array = [[float(1) for _ in range(self.sol_shape[1])] for _ in range(self.sol_shape[0])]
        print(f'Phaeremone array: {self.pheromone_array}')
        while self.patience_count < self.patience:
            self.next()
            print('='*50)
        print("Best Solution:")
        self.BestSolution.output()

    def calc_probability(self):
        self.proability_array = []
        for n in self.pheromone_array: 
            k_tau = sum(n)
            L2 = [(np.float_power(x,self.a))/k_tau for x in n]
            self.proability_array.append(L2)
            
    def evaporate(self):
        for p1 in self.pheromone_array:
            for p2 in p1: 
                p2 = (1-self.evapaporation)*p2
    
    def update_pheremone(self): 
        for p1 in range(len(self.pheromone_array)):
            for p2 in range(len(self.pheromone_array[p1])): 
                for ants in self.ants: 
                    #print(ants.solution_matrix)
                    self.pheromone_array[p1][p2] += ants.solution_matrix[p1][p2]*self.pheromones/ants.cost

    def next(self):
        self.calc_probability()
        newBestCost = float('inf')
        avg_total_violation = 0 
        avg_skill_violation = 0 
        avg_skill_lvl_violation = 0 
        avg_deadline_violation = 0 
        avg_overtime_violation = 0
        for A in self.ants:
            print(f'Ant Solution Matrix Before: {A.solution_matrix}')
            print(f'Probability Array: {self.proability_array}') 
            A.update(self.proability_array)

            avg_total_violation += sum([A.skill_lvl_violation,A.skill_violation,A.deadline_violation,A.overtime_violation])
            avg_skill_lvl_violation += A.skill_lvl_violation
            avg_skill_violation += A.skill_violation 
            avg_deadline_violation += A.deadline_violation 
            avg_overtime_violation += A.overtime_violation 

            if A.cost <= newBestCost:
                newBestCost = A.cost
                newBestSolution = A 
            print(f'Ant Solution Matrix After: {A.solution_matrix}')
        self.update_pheremone()
        print(f'After Pheremone update {self.pheromone_array}\n')
        self.evaporate()
        print(f'After evaporation: {self.pheromone_array}\n')


        avg_total_violation = avg_total_violation/len(self.ants) 
        avg_skill_lvl_violation = avg_skill_lvl_violation/len(self.ants)
        avg_skill_violation = avg_skill_violation/len(self.ants) 
        avg_deadline_violation = avg_deadline_violation/len(self.ants) 
        avg_overtime_violation = avg_overtime_violation/len(self.ants) 

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
        

    def plot_cost(self):
        plt.subplot(3,1,1)
        plt.plot(self.cost_history,'b-',linewidth=3,label ='Best Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.subplot(3,1,2)
        plt.plot(self.averagetotalViolatioHist,'r-',linewidth=3,label='total avg Violations')
        plt.xlabel('Iterations')
        plt.ylabel("# of violations")
        plt.subplot(3,1,3)
        plt.plot(self.skill_lvl_violationHist,'o-',linewidth= 2,label='skill lvl violations')
        plt.plot(self.skill_violationHist,'g-',linewidth = 2, label= 'Skill Violations')
        plt.plot(self.deadline_violationHist,'r-',linewidth = 2 , label='Deadline Violation')
        plt.plot(self.overtime_violationHist,'b-',linewidth = 2 , label = 'Overtime Violation')
        plt.xlabel('Iterations')
        plt.ylabel('# of violations')
        plt.show()


# Ant_employees, Ant_Tasks = DS.Generate_data(['A','B','C','D','E'],10,25)
# A = AntColonyOptimser(5,1,0.8,0.02,Ant_employees,Ant_Tasks,patience=100)

# A.plot_cost()