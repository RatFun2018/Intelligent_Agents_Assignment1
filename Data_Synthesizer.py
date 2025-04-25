import numpy 
import random as rd

def Generate_data(availableskills,numemployees=10,numTasks=15):
    Employees = {} 
    Tasks = {}
    emnum = 1
    for e in range(numemployees):
        newEmployee = {"Hours": rd.randint(1,24),"Skill_lvl":rd.randint(1,5),"Skills":rd.sample(availableskills,len(availableskills)),"Assigned Tasks":{}}
        Ekey = 'E' + str(emnum)
        Employees.update({Ekey:newEmployee})
        emnum+= 1
    tnum = 1
    for t in range(numTasks): 
        newTask = {"Estimated Time":2,"Difficulty":2,"Deadline":5,"Skills":'A'}
        Tkey = 'T' + str(tnum)
        Tasks.update({Tkey:newTask})
        tnum+=1

    return Employees,Tasks

E,T = Generate_data(['A','B','C','D','E'])
print(E)
print(T)
    