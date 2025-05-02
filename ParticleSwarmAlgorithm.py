import numpy as np
import pandas as pd
import random as rd
import Data_Synthesizer as DS 
import math
import matplotlib.pyplot as plt 
import psutil
import time
import os


class Particle:
  '''
    Particle object 
    contains attributes and methods to function as a particle with a individual solution in a particle swarm optmiser 
    Initialisation: 
    Input:

    format for employee and task information is outlied in the data systhesizer method 
    employees: list type object containing employee information 
    tasks: list type object containing task information 
  '''

  def __init__(self,Employees,Tasks):
    self.Employees= Employees
    self.Tasks = Tasks
    
    self.skill_lvl_violation = 0  #Variable to keep track of skill level violations in particles solution 
    self.skill_violation =  0     #Variable to keep track of skill violations in particles solution 
    self.overtime_violation = 0   #Variable to keep track of overtime violations in particles solution 
    self.deadline_violation = 0   #Variable to keep track of deadline violation sin particles solution 
    self.cost = float("inf")      #Intialising cost funtion 
    #Solution matrix and velocity matrix to determine the change in the solution 
    self.solution_matrix = [[ 0 for _ in range(len(self.Employees))] for _ in range(len(self.Tasks))] #Intialises solution matrix as a 2d matrix based on the number of employees and tasks in the problem 
    for T in self.solution_matrix: #Loops through all rows in the solution matrix which represents the tasks 
      c = rd.randint(0,len(T)-1) # randomly chooses an employee for the task to be assigned too
      T[c] = 1
    self.solution_velocity_matrix = [[rd.uniform(-1,1) for _ in range(len(self.Employees))] for _ in range(len(self.Tasks))] # intialises the velocity matrix of the particle as an array of values between -1-1 
    self.pBest = self.solution_matrix # Intialises  the particles best seen solution as the intial solution 
    self._translate_solution()

  def _translate_solution(self):
    '''
        Function that translates the solution binary matrix into a Dictionary Format of displaying the task assignmnet information 
        Implementation of the cost calculation was easier with a dictionary data format but other functions such as pheremone calculation 
        and path determination was better done in a matrix format 
    '''
    self.Employees_Assigned =self.Employees

    #Resets the metric tracking so that the metrics are based on the current solution 
    self.skill_lvl_violation = 0 
    self.skill_violation =  0
    self.overtime_violation = 0
    self.deadline_violation = 0


    for f in self.Employees_Assigned: #Wipes Tasks Assignments to not carry over previous solution 
      f['Assigned Tasks'] = {}
    task_idx = 0
    for T in self.solution_matrix: #iterates through all the tasks in the solution matrix 
      employee_idx = 0 #indexing value that is used to match the value in the solution matrix to the correct employee in the dictionary representation 
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
          not_skill += 1

        if E['Assigned Tasks'][T]['Difficulty'] > E['Skill_lvl']:
          skilldiff += 1

        over_Deadline = max(cumualitive_tasktime-E['Assigned Tasks'][T]['Deadline'],0)
        if over_Deadline != 0: 
          self.deadline_violation += 1 
      #print(f'cumulative Task Time: {cumualitive_tasktime}')
      overtime = max(cumualitive_tasktime-E['Hours'],0)
      if overtime != 0: 
        self.overtime_violation +=1 
      newcost += (0.25 * not_skill + 0.25 * skilldiff + 0.25 * over_Deadline + 0.25 * overtime)
      self.skill_lvl_violation = skilldiff 
      self.skill_violation = not_skill 
      
    if newcost < self.cost:
      self.pBest = self.solution_matrix
    self.cost = newcost






class Particle_Swarm_Optimiser:
  '''
  Particle Swarm Optimiser for task assignment 
  Intialisiation
  Input: 
  n_particles: Number of particles to be generated to optimse 
  w: weight value that influences how much the current velocity of the particle will effect the new velocity 
  c1: value that influences how much the global best value changes the particle velocity 
  c2: value that influences how much the particles best value chnage the particle velocity 
  
  format for employee and task information is outlied in the data systhesizer method 
  employees: list type object containing employee information 
  tasks: list type object containing task information 

  n_iter: number of iterations that the optimiser will run 
    

  '''
  def __init__(self, n_particles, w, c1, c2,Employees,Tasks,n_iter=10,patience=3):
    self.n_particles = n_particles
    self.particles = []
    self.patience = patience
    self.n_iter =n_iter
    self.w = w
    self.c1 = c1
    self.c2 = c2

    # Variables to track the global best solution in all particles and log the global best over all iterations 
    self.gBest = None 
    self.gBest_cost = float('inf')
    self.gBestHistory = []
    self.gBestCostHistory = []

    #Keeping track of metrics to be graphed at the end of optmisation process 
    self.averagetotalViolatioHist = []          #Tracks average total violation in each Iteration 
    self.skill_violationHist = []               #Tracks average skill violations in each Iteration 
    self.skill_lvl_violationHist = []           #Tracks average skill level violation in each Iteration
    self.deadline_violationHist = []            #Tracks average amount of  deadline violation in each Iteration
    self.overtime_violationHist = []            #Tracks average amount of ovetime violation in each Iteration
    self.process_timeHist = []                  #Tracks process time taken in Iterations
    self.memoryuseHist = []                     #Tracks memeory usage in Iterations
    self.process = psutil.Process(os.getpid())  #gets the running python process to get memory usage data 
    
    self.generate_particles(Employees,Tasks) #Function that generates the required amount of particles using the given employee and task data 

    for i in range(self.n_iter): #Runs the optmiser the specified amount of runs 
      print(f' Step {i}')
      print('='*20)
      self.next()
      self.gBestCostHistory.append(self.gBest_cost) #logging for best global cost for every iteration 
      #if self.check_termination():
        #break
      print('='*20)


  def generate_particles(self,Employees,Tasks):
    '''
      Function that generates the required amount of particles using the given employee and task data 
    '''
    for n in range(self.n_particles):
      new_particle = Particle(Employees,Tasks)
      self.particles.append(new_particle)
  
  def check_termination(self):
    '''
      function that checks if optimiser should stop if there has been no change in the best value for a certain number of iterations 
      currently not implemented 
    '''
    if len(self.gBestHistory) >= self.patience:
      improvement = abs(self.gBestHistory[-1] - self.gBestHistory[-self.patience])
      if improvement < 0.001:
        return True 


  def next(self):
    '''
    Main Funtion that moves the optimisation forward 
    '''


    avg_total_violation = 0             #variable to log the avg total violations each iteration 
    avg_skill_violation = 0             #variable to log the avg skill matching violations each iteration
    avg_skill_lvl_violation = 0         #variable to log the avg skill lvl violations each iteration
    avg_deadline_violation = 0          #variable to log the avg deadline violations each iteration
    avg_overtime_violation = 0          #variable to log the avg overtime violations each iteration
    start_time = time.time()            #take time at start of process for time performance tracking 
    start_mem = self.process.memory_info().rss / 1024 


    #print(avg_total_violation) 
    for k in self.particles:
      #k.output()
      k.update_particle() #updating the 'position' of the particle which means possibly changing the solution of  the particle based on its velocity values 
      print(f'cost: {k.cost}\n')

      #Variables are added to get the cumulative values for each valuation over all partilces in the iteration 
      avg_total_violation += sum([k.skill_lvl_violation,k.skill_violation,k.deadline_violation,k.overtime_violation])
      avg_skill_lvl_violation += k.skill_lvl_violation
      avg_skill_violation += k.skill_violation 
      avg_deadline_violation += k.deadline_violation 
      avg_overtime_violation += k.overtime_violation 


      if k.cost <= self.gBest_cost: #Checks if the current particles cost is the best cost value seen so far 
        self.gBest_cost = k.cost
        self.gBest = k.pBest
        print(f'New gBest: {self.gBest}')
    self.gBestHistory.append(self.gBest) #Logging for Change in best cost over iteration 
    #print(avg_total_violation)

    #All the cumulative values are devided by the amount of particles to get an average value of the measurement metrics 
    avg_total_violation = avg_total_violation/len(self.particles) 
    avg_skill_lvl_violation = avg_skill_lvl_violation/len(self.particles)
    avg_skill_violation = avg_skill_violation/len(self.particles) 
    avg_deadline_violation = avg_deadline_violation/len(self.particles) 
    avg_overtime_violation = avg_overtime_violation/len(self.particles) 
    
    #Appending all the metrics tracked in this iteration into a list to be graphed at the end 
    self.averagetotalViolatioHist.append(avg_total_violation) 
    self.skill_lvl_violationHist.append(avg_skill_lvl_violation)
    self.skill_violationHist.append(avg_skill_violation) 
    self.deadline_violationHist.append(avg_deadline_violation)
    self.overtime_violationHist.append(avg_overtime_violation)
    
    for j in self.particles: #Iterates through all partciles 
      j.update_velocity(self.gBest,self.w,self.c1,self.c2) #Particle velocity is updated done after all positions of the partilces have updated 
    

    #Time taken and memory usage metric tracking and logging 
    end_time = time.time()
    end_mem =  self.process.memory_info().rss / 1024
    mem_used = end_mem - start_mem
    iteration_time  = end_time - start_time
    self.memoryuseHist.append(mem_used)
    self.process_timeHist.append(iteration_time)
  
  def plot_cost(self):

    '''
        Simple quick and dirty graphing function that tries to display or logged metrics over the number of iterations of the optimiser 
    '''
    plt.subplot(3,2,1)
    plt.plot(self.gBestCostHistory,'b-',linewidth=3,label ='Best Fitness')
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


# E,T = DS.Generate_data(['A','B','C','D','E','F','G'],10,18)
# Swarm = Particle_Swarm_Optimiser(25,0.3,2,3,E,T,n_iter=500)
# Swarm.plot_cost()
# print(f'gBest = {Swarm.gBest}')