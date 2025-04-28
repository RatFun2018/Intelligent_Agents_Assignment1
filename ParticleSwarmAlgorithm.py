import numpy as np
import pandas as pd
import random as rd
import Data_Synthesizer as DS 
import math
import matplotlib.pyplot as plt 

E1={"Hours": 10,"Skill_lvl":3,"Skills":['A','C'],"Assigned Tasks":{}}
E2={"Hours": 8,"Skill_lvl":2,"Skills":['B','C'],"Assigned Tasks":{}}
E3={"Hours": 12,"Skill_lvl":3,"Skills":['C','B'],"Assigned Tasks":{}}
Employees = [E1,E2,E3]
T1={"Estimated Time":2,"Difficulty":2,"Deadline":5,"Skills":'A'}
T2={"Estimated Time":3,"Difficulty":2,"Deadline":7,"Skills":'C'}
T3={"Estimated Time":4,"Difficulty":2,"Deadline":4,"Skills":'B'}
Tasks = [T1,T2,T3]
class Particle:
  def __init__(self,Employees,Tasks):
    self.Employees= Employees
    self.Tasks = Tasks
    
    self.cost = float("inf")
    #Solution matrix and velocity matrix to determine the change in the solution 
    self.solution_matrix = [[ 0 for _ in range(len(self.Employees))] for _ in range(len(self.Tasks))]
    for T in self.solution_matrix:
      c = rd.randint(0,len(T)-1)
      T[c] = 1
    self.solution_velocity_matrix = [[rd.uniform(-1,1) for _ in range(len(self.Employees))] for _ in range(len(self.Tasks))]
    self.pBest = self.solution_matrix
    self._translate_solution()

  def _translate_solution(self):
    self.Employees_Assigned =self.Employees
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

    for E in self.Employees_Assigned:
      sorted_tasks = dict(sorted(E['Assigned Tasks'].items(),key=lambda x: (x[1]['Deadline'],x[1]['Estimated Time'])))
      E['Assigned Tasks'] = sorted_tasks

    self.fitness()

  def _activation(self,x):
    out = 1/(1+math.exp(-x))
    
    return out 

  def roulette_selection(self,probabilities):
    total = sum(probabilities)
    normalise_probs = [p/total for p in probabilities]

    
    #print(f'Normalised: {normalise_probs}')

    r = rd.random()
    
    total_probs = 0.0 
    out = len(normalise_probs) - 1 
    for j in range(len(normalise_probs)):
      prob = normalise_probs[j]
      total_probs += prob
      #print(f'{r},{total_probs}')
      if r <= total_probs: 
        out = j 
        break
    
    return out 
    


  def update_particle(self):
    new_pos = [[ 0 for _ in range(len(self.Employees))] for _ in range(len(self.Tasks))]
    for k in range(len(self.Tasks)):
      probs = [self._activation(self.solution_velocity_matrix[k][j]) for j in range(len(self.Employees))]
      #print(probs)
      j_selected = self.roulette_selection(probs)
      #print(j_selected)

      new_pos[k][j_selected] = 1 
    

    self.solution_matrix = new_pos  
    self._translate_solution()
    
  def update_velocity(self,gBest,w,c1,c2):
    for i in range(len(self.Tasks)):
      for j in range(len(self.Employees)):
        cog = c1 * rd.random() * (self.pBest[i][j] - self.solution_matrix[i][j])
        soc = c2 * rd.random() * (gBest[i][j] - self.solution_matrix[i][j])

        self.solution_velocity_matrix[i][j] = w * self.solution_velocity_matrix[i][j] + cog + soc 

        #Clamping the velocity values 
        clamp = 5
        self.solution_velocity_matrix[i][j] = np.clip(self.solution_velocity_matrix[i][j],-clamp,clamp)

  def output(self):
    E_idx = 0
    for E in self.Employees_Assigned:
      e_name = 'E' + str(E_idx)
      print(f'{e_name}: {E}\n')
      E_idx += 1
    print(f'Solution Matrix: {self.solution_matrix}\n')
    print(f'Velocity Matrix: {self.solution_velocity_matrix}\n')
    print(f'Cost: {self.cost}')

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
      #print(f'cumulative Task Time: {cumualitive_tasktime}')
      overtime = max(cumualitive_tasktime-E['Hours'],0)
      newcost += (0.25 * not_skill + 0.25 * skilldiff + 0.25 * over_Deadline + 0.25 * overtime)
    if newcost < self.cost:
      self.pBest = self.solution_matrix
    self.cost = newcost






class Particle_Swarm_Optimiser:
  def __init__(self, n_particles, w, c1, c2,n_iter=10,patience=3):
    self.n_particles = n_particles
    self.particles = []
    self.patience = patience
    self.n_iter =n_iter
    self.w = w
    self.c1 = c1
    self.c2 = c2
    self.gBest = None 
    self.gBest_cost = float('inf')
    self.gBestHistory = []
    self.gBestCostHistory = []

    self.generate_data()

    for i in range(self.n_iter):
      print(f' Step {i}')
      print('='*20)
      self.next()
      self.gBestCostHistory.append(self.gBest_cost)
      #if self.check_termination():
        #break
      print('='*20)


  def generate_data(self):
    _Employees,_Tasks = DS.Generate_data(['A','B','C','D','E'],10,25)
    for n in range(self.n_particles):
      new_particle = Particle(_Employees,_Tasks)
      self.particles.append(new_particle)
  
  def check_termination(self):
    if len(self.gBestHistory) >= self.patience:
      improvement = abs(self.gBestHistory[-1] - self.gBestHistory[-self.patience])
      if improvement < 0.001:
        return True 


  def next(self):
    
    for k in self.particles:
      k.output()
      k.update_particle()
      print(f'cost: {k.cost}\n')
      if k.cost <= self.gBest_cost:
        self.gBest_cost = k.cost
        self.gBest = k.pBest
        print(f'New gBest: {self.gBest}')
    self.gBestHistory.append(self.gBest)
    
    for j in self.particles:
      j.update_velocity(self.gBest,self.w,self.c1,self.c2) 
  
  def plot_cost(self):
    plt.plot(self.gBestCostHistory,'b-',linewidth=3,label ='Best Fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()



Swarm = Particle_Swarm_Optimiser(25,0.3,2,3,n_iter=25)
Swarm.plot_cost()
print(f'gBest = {Swarm.gBest}')