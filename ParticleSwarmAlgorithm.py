import numpy as np
import pandas as pd
import random as rd
E1={"Hours": 10,"Skill_lvl":4,"Skills":['A','B','C'],"Assigned Tasks":{}}
E1={"Hours": 12,"Skill_lvl":2,"Skills":['B','C'],"Assigned Tasks":{}}
Employees = {'E1':E1,'E2':E2}
T1={"Estimated Time":2,"Difficulty":2,"Deadline":5,"Skills":'A'}
T2={"Estimated Time":3,"Difficulty":2,"Deadline":7,"Skills":'C'}
T3={"Estimated Time":4,"Difficulty":2,"Deadline":4,"Skills":'B'}
Tasks = {'T1':T1,'T2':T2,'T3':T3}
class Particle:
  def __init__(self,Employees,Tasks):
    self.Employees= Employees
    self.Tasks = Tasks
    self.cost = 0
    self.mutate()


  def mutate(self):
    new_Employees = self.Employees
    for T in self.Tasks:
      td = self.Tasks[T]
      choice = rd.choice(list(self.Employees.values()))
      choice['Assigned Tasks'].update({T:td})
      #for E in self.Employees.values():
        #if td['Skills'] in E['Skills'] and td['Estimated Time'] < E['Hours'] and td['Difficulty'] <= E['Skill_lvl']:
          #shortlist.append(E)
        #choice = rd.randint(0,len()-1)
        #shortlist[choice]['Assigned Tasks'].append(T)
        #shortlist[choice]['Hours'] -= td['Estimated Time']
    for E in new_Employees.values():
      sorted_tasks = dict(sorted(E['Assigned Tasks'].items(),key=lambda x: (x[1]['Deadline'],x[1]['Estimated Time'])))
      E['Assigned Tasks'] = sorted_tasks

    self.Assigned_Employees = new_Employees
    self.fitness()

  def output(self):
    for E in self.Employees.values():
      print(E)

  def fitness(self):
    newcost = 0
    for E in self.Assigned_Employees.values():
      cumualitive_tasktime = 0
      not_skill = 0
      skilldiff =0
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

  def next(self):
    ...


p = Particle(Employees,Tasks)
p.output()
p.fitness()