import numpy as np
import pandas as pd
import random as rd
E1={"Hours": 10,"Skill_lvl":4,"Skills":['A','B','C'],"Assigned Tasks":{}}
E2={"Hours": 12,"Skill_lvl":2,"Skills":['B','C'],"Assigned Tasks":{}}
Employees = [E1,E2]
T1={"Estimated Time":2,"Difficulty":2,"Deadline":5,"Skills":'A'}
T2={"Estimated Time":3,"Difficulty":2,"Deadline":7,"Skills":'C'}
T3={"Estimated Time":4,"Difficulty":2,"Deadline":4,"Skills":'B'}
Tasks = [T1,T2,T3]
class Particle:
  def __init__(self,Employees,Tasks):
    self.Employees= Employees
    self.Tasks = Tasks
    
    self.cost = 0
    #Solution matrix and velocity matrix to determine the change in the solution 
    self.solution_matrix = [[ rd.randint(0,1) for _ in range(len(self.Tasks))] for _ in range(len(self.Employees))]
    self.solution_velocity_matrix = [[rd.randint(0,1) for _ in range(len(self.Tasks))] for _ in range(len(self.Employees))]

    self._translate_solution()

  def _translate_solution(self):
    self.Employees_Assigned = self.Employees
    employee_idx = 0
    for E in self.solution_matrix:
      task_idx = 0
      for T in E: 
        print(T)
        if T == 1:
          print(f'task_idx: {task_idx}')
          taskname = 'T' + str(task_idx)
          self.Employees_Assigned[employee_idx]['Assigned Tasks'].update({taskname:self.Tasks[task_idx]})
        task_idx +=1
      employee_idx +=1

    for E in self.Employees:
      sorted_tasks = dict(sorted(E['Assigned Tasks'].items(),key=lambda x: (x[1]['Deadline'],x[1]['Estimated Time'])))
      E['Assigned Tasks'] = sorted_tasks

    self.fitness()

  def mutate(self):
    for k in range(len(self.solution_velocity_matrix)):
      for j in range(len(self.solution_velocity_matrix[k])):
        self.solution_matrix[k][j] = self.solution_matrix[k][j] ^ self.solution_velocity_matrix[k][j]

    
    self._translate_solution()
    

  def output(self):
    for E in self.Employees:
      print(E)
    print(f'Solution Matrix: {self.solution_matrix}\n')
    print(f'Velocity Matrix: {self.solution_velocity_matrix}\n')
    print(f'Cost: {self.cost}')

  def fitness(self):
    newcost = 0
    for E in self.Employees:
      cumualitive_tasktime = 0
      not_skill = 0
      skilldiff =0
      over_Deadline = 0
      for T in E['Assigned Tasks']:
        cumualitive_tasktime += E['Assigned Tasks'][T]['Estimated Time']
        if E['Assigned Tasks'][T]['Skills'] not in E['Skills']:
          not_skill = 100

        if E['Assigned Tasks'][T]['Difficulty'] > E['Skill_lvl']:
          skilldiff = 100

        over_Deadline = max(cumualitive_tasktime-E['Assigned Tasks'][T]['Deadline'],0)

      overtime = max(cumualitive_tasktime-E['Hours'],0)
      newcost = 0.2 * not_skill + 0.2 * skilldiff + 0.4 * over_Deadline + 0.2 * overtime
      self.velocity = self.cost - newcost
      self.cost = newcost






class Particle_Swarm_Optimiser:
  def __init__(self, n_particles, n_dimensions, bounds, w, c1, c2):
    self.n_particles = n_particles
    self.n_dimensions = n_dimensions
    self.bounds = bounds
    self.w = w
    self.c1 = c1
    self.c2 = c2

  def next(self):
    ...


p = Particle(Employees,Tasks)
p.output()
p.mutate()
p.output()