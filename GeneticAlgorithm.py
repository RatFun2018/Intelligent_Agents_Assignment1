#Code for The Genetic Algor
'''
Genetic Algorithm (GA):
– Encoding: Represent each solution as a vector where each element indicates the employee assigned to a specific task.
– Operators: Use crossover and mutation to evolve the solution population.
– Fitness: Evaluate solutions based on penalties for any constraint violations

'''

# #Sample Employee & Task Data from Assignment Brief
# E1 = {"Hours": 10, "Skill_lvl": 4, "Skills": ['A', 'C'], "Assigned Tasks": {}}
# E2 = {"Hours": 12, "Skill_lvl": 6, "Skills": ['A', 'B', 'C'], "Assigned Tasks": {}}
# E3 = {"Hours": 8, "Skill_lvl": 3, "Skills": ['A'], "Assigned Tasks": {}}
# E4 = {"Hours": 15, "Skill_lvl": 7, "Skills": ['B', 'C'], "Assigned Tasks": {}}
# E5 = {"Hours": 9, "Skill_lvl": 5, "Skills": ['A', 'C'], "Assigned Tasks": {}}

# Employees = [E1, E2, E3, E4, E5]

# T1 = {"Estimated Time": 4, "Difficulty": 3, "Deadline": 8, "Skills": 'A'}
# T2 = {"Estimated Time": 6, "Difficulty": 5, "Deadline": 12, "Skills": 'B'}
# T3 = {"Estimated Time": 2, "Difficulty": 2, "Deadline": 6, "Skills": 'A'}
# T4 = {"Estimated Time": 5, "Difficulty": 4, "Deadline": 10, "Skills": 'C'}
# T5 = {"Estimated Time": 3, "Difficulty": 1, "Deadline": 7, "Skills": 'A'}
# T6 = {"Estimated Time": 8, "Difficulty": 6, "Deadline": 15, "Skills": 'B'}
# T7 = {"Estimated Time": 4, "Difficulty": 3, "Deadline": 9, "Skills": 'C'}
# T8 = {"Estimated Time": 7, "Difficulty": 5, "Deadline": 14, "Skills": 'B'}
# T9 = {"Estimated Time": 2, "Difficulty": 2, "Deadline": 5, "Skills": 'A'}
# T10 = {"Estimated Time": 6, "Difficulty": 4, "Deadline": 11, "Skills": 'C'}

# Tasks = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]
import numpy as np
import pandas as pd
import random as rd
import Data_Synthesizer as DS
import math
import matplotlib.pyplot as plt
import copy
import psutil
import time

# Penalty Weights (from Assignment Brief)
ALPHA = 0.20  # Overload penalty
BETA  = 0.20  # Skill mismatch penalty
DELTA = 0.20  # Difficulty violation penalty
GAMMA = 0.20  # Deadline violation penalty
SIGMA = 0.20  # Unique assignment penalty (not used in this encoding)

# GAIndividual: Represents one solution
class GAIndividual:
    def __init__(self, employees, tasks):
        self.employees = employees
        self.tasks = tasks
        self.assignment = [rd.randint(0, len(employees) - 1) for _ in range(len(tasks))]
        self.cost = self.calculate_cost()

    def calculate_cost(self):
        employees = self.employees
        cost = 0

        # Reset task assignments
        for e in employees:
            e['Assigned Tasks'] = {}

        for i, emp_id in enumerate(self.assignment):
            task_name = f"T{i}"
            employees[emp_id]['Assigned Tasks'][task_name] = self.tasks[i]

        for emp in employees:
            assigned = list(emp['Assigned Tasks'].items())
            assigned.sort(key=lambda x: (x[1]['Deadline'], x[1]['Estimated Time']))
            time_used = 0
            penalty_skill = 0
            penalty_difficulty = 0
            penalty_deadline = 0

            for _, task in assigned:
                time_used += task['Estimated Time']
                if task['Skills'] not in emp['Skills']:
                    penalty_skill += 1
                if task['Difficulty'] > emp['Skill_lvl']:
                    penalty_difficulty += 1
                penalty_deadline += max(0, time_used - task['Deadline'])

            penalty_overload = max(0, time_used - emp['Hours'])

            cost += (
                ALPHA * penalty_overload +
                BETA  * penalty_skill +
                DELTA * penalty_difficulty +
                GAMMA * penalty_deadline
            )

        return cost

    def mutate(self, mutation_rate=0.1):
        for i in range(len(self.assignment)):
            if rd.random() < mutation_rate:
                self.assignment[i] = rd.randint(0, len(self.employees) - 1)

    def crossover(self, other):
        point = rd.randint(1, len(self.assignment) - 1)
        child1 = GAIndividual(self.employees, self.tasks)
        child2 = GAIndividual(self.employees, self.tasks)
        child1.assignment = self.assignment[:point] + other.assignment[point:]
        child2.assignment = other.assignment[:point] + self.assignment[point:]
        child1.cost = child1.calculate_cost()
        child2.cost = child2.calculate_cost()
        return child1, child2

# Genetic Algorithm Optimiser
class GeneticAlgorithm:
    def __init__(self, employees, tasks, pop_size=10, generations=20, mutation_rate=0.1):
        self.employees = employees
        self.tasks = tasks
        self.population = [GAIndividual(employees, tasks) for _ in range(pop_size)]
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best = min(self.population, key=lambda x: x.cost)
        self.best_costs = []
        self.memory_usage = []  # Track memory usage per generation
        self.constraint_violations = []  # Track constraint violations per generation
        self.elapsed_times = []  # Track elapsed time per generation
        

    def evolve(self):
        for gen in range(self.generations):
            start_time = time.time()  # Start time for the current generation
            # Track absolute memory usage at the start of the generation
            start_memory = psutil.Process().memory_info().rss / 1024  # Memory usage in KB at the start

            new_population = []
            self.population.sort(key=lambda x: x.cost)

            # Elitism: carry over best 2
            new_population.extend(self.population[:2])

            # Fill rest of population
            while len(new_population) < len(self.population):
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = parent1.crossover(parent2)
                child1.mutate(self.mutation_rate)
                child2.mutate(self.mutation_rate)
                new_population.extend([child1, child2])

            self.population = new_population[:len(self.population)]
            current_best = min(self.population, key=lambda x: x.cost)
            if current_best.cost < self.best.cost:
                self.best = current_best

            # Track performance for the current generation
            self.best_costs.append(self.best.cost)

            # Track memory usage for the current generation
            current_memory_usage = psutil.Process().memory_info().rss / 1024  # Absolute memory usage (in KB)
            self.memory_usage.append(current_memory_usage)  # Record memory usage for this generation

            # Track constraint satisfaction (total penalty)
            total_violations = sum([ind.cost for ind in self.population])  # Sum of penalties for all individuals
            self.constraint_violations.append(total_violations)

            # Track elapsed time for the current generation
            end_time = time.time()  # End time for the current generation
            elapsed_time = end_time - start_time  # Time taken for this generation
            self.elapsed_times.append(elapsed_time)

            print(f"Generation {gen + 1} - Best Cost: {self.best.cost:.2f}")

    def tournament_selection(self, k=3):
        return min(rd.sample(self.population, k), key=lambda x: x.cost)

    # Plotting Solution Quality (Optimality) - Objective Function (Total Penalty)
    def plot_cost(self):
        plt.subplot(1, 3, 1)  # Row 1, Column 1
        plt.plot(self.best_costs, color='blue', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Objective (Cost) Function Value (Total Penalty)')
        plt.title('Solution Quality (Optimality)')
        plt.grid(True)

    # Plotting Computational Efficiency (Memory Usage)
    def plot_memory_usage(self):
        plt.subplot(1, 3, 2)  # Row 1, Column 2
        plt.plot(self.memory_usage, color='orange', linewidth=2)  # Track memory usage every generation
        plt.xlabel('Generation')
        plt.ylabel('Memory Usage (KB)')
        plt.title('Computational Efficiency (Memory Usage)')
        plt.grid(True)

    # Plotting Constraint Satisfaction (Feasibility) and Elapsed Time (with dual-axis)
    def plot_constraint_violations(self):
        plt.subplot(1, 3, 3)  # Row 1, Column 3
        ax1 = plt.gca()  # Get the current axis for the plot

        # Plot Constraint Violations (Primary Y-axis)
        ax1.plot(range(0, self.generations, 10), self.constraint_violations[::10], color='red', linewidth=2)  # Downsample every 10th generation
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Total Constraint Violations/Generation', color='red')
        ax1.tick_params(axis='y', labelcolor='red')

        # Create a second Y-axis for Elapsed Time
        ax2 = ax1.twinx()
        ax2.plot(range(0, self.generations, 10), self.elapsed_times[::10], color='green', linewidth=2)  # Downsample every 10th generation
        ax2.set_ylabel('Elapsed Time/Generation (seconds)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Adding title for Elapsed Time axis (Green)
        ax2.set_title('Constraint Satisfaction (Feasibility) and Elapsed Time Over Generations')

        plt.grid(True)




# # Importing the Data Synthesizer and generating synthetic data
# GA_Employees, GA_Tasks = DS.Generate_data(['A','B','C','D','E'], 5, 10)

# # Set up the GA for 500 generations
# GA = GeneticAlgorithm(GA_Employees, GA_Tasks, pop_size=20, generations=500, mutation_rate=0.2)

# # Evolve the GA to calculate results
# GA.evolve()

# # Now the GA is evolved, so you can safely print the best assignment and best cost
# print(f"\nBest assignment (task → employee): {GA.best.assignment}")
# print(f"Best cost: {GA.best.cost:.2f}")

# # Plotting the results
# plt.figure(figsize=(18, 6))  # Increase figure size to fit all 3 plots side by side
# GA.plot_cost()  # Solution Quality (Objective Function)
# GA.plot_memory_usage()  # Computational Efficiency (Memory Usage)
# GA.plot_constraint_violations()  # Constraint Satisfaction (Feasibility) and Elapsed Time

# # Adjusting spacing between subplots
# plt.subplots_adjust(wspace=0.3)  # Increase space between subplots to avoid overlap

# plt.tight_layout()  # Ensures everything fits inside the plot
# plt.show()
