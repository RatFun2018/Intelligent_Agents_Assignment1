import numpy 
import random as rd

def Generate_data(availableskills,numemployees=10,numTasks=15):
    '''
    Generates synthetic data to test Optimiser functions 
    Input: list of available skills eg.['A','B','C'], number of employees to generate, number of Tasks to generate 
    Output: List of Employees where each employee is a dictionary containing "Hours",'Skill_lvl",Skkls & List of Tasks 
    '''
    Employees = [] 
    Tasks = []
    emnum = 1
    for e in range(numemployees):
        newEmployee = {"Hours": rd.randint(1,24),"Skill_lvl":rd.randint(1,6),"Skills":rd.sample(availableskills,rd.randint(1,len(availableskills))),"Assigned Tasks":{}}
        Ekey = 'E' + str(emnum)
        Employees.append(newEmployee)
        emnum+= 1
    tnum = 1
    for t in range(numTasks): 
        newTask = {"Estimated Time":rd.randint(1,8),"Difficulty":rd.randint(1,6),"Deadline":rd.randint(1,12),"Skills":rd.choice(availableskills)}
        Tkey = 'T' + str(tnum)
        Tasks.append(newTask)
        tnum+=1

    return Employees,Tasks

#E,T = Generate_data(['A','B','C','D','E'])
#print(E)
#print(T)
    