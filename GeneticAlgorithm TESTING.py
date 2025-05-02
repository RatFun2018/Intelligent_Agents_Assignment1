#Code for The Genetic Algor
'''
Genetic Algorithm (GA):
– Encoding: Represent each solution as a vector where each element indicates the employee assigned to a specific task.
– Operators: Use crossover and mutation to evolve the solution population.
– Fitness: Evaluate solutions based on penalties for any constraint violations

'''
import numpy as np
import pandas as pd
import random as rd
import Data_Synthesizer as DS
import math
import matplotlib.pyplot as plt
import copy
import time
import psutil

# Penalty Weights (from Assignment Brief)
ALPHA = 0.20  # Overload penalty
BETA  = 0.20  # Skill mismatch penalty
DELTA = 0.20  # Difficulty violation penalty
GAMMA = 0.20  # Deadline violation penalty
SIGMA = 0.20  # Unique assignment penalty (not used in this encoding)

# GAIndividual: Represents one solution
class GAIndividual:
    def __init__(self, employees, tasks):
        self.employees = copy.deepcopy(employees)
        self.tasks = tasks
        self.assignment = [rd.randint(0, len(employees) - 1) for _ in range(len(tasks))]
        self.cost = self.calculate_cost()

    def calculate_cost(self):
        employees = copy.deepcopy(self.employees)
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
        self.runtime = []  # Track runtime per generation
        self.memory_usage = []  # Track memory usage per generation
        self.constraint_violations = []  # Track constraint violations per generation

    def evolve(self):
        for gen in range(self.generations):
            start_time = time.time()  # Start time for the current generation
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

            # Track computational efficiency for the current generation
            end_time = time.time()  # End time for the current generation
            elapsed_time = end_time - start_time  # Time taken for this generation
            current_memory_usage = (psutil.Process().memory_info().rss / 1024) - start_memory  # Memory in KB
            self.runtime.append(elapsed_time)
            self.memory_usage.append(current_memory_usage)

            # Track constraint satisfaction (total penalty)
            total_violations = sum([ind.cost for ind in self.population])  # Sum of penalties for all individuals
            self.constraint_violations.append(total_violations)

            print(f"Generation {gen + 1} - Best Cost: {self.best.cost:.2f}")

    def tournament_selection(self, k=3):
        return min(rd.sample(self.population, k), key=lambda x: x.cost)

    def plot_cost(self):
        plt.plot(self.best_costs, color='blue', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best Cost')
        plt.title('GA Convergence (Solution Quality)')
        plt.grid(True)
        plt.show()

    def plot_runtime(self):
        plt.plot(self.runtime, color='green', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Runtime (seconds)')
        plt.title('Computational Efficiency (Runtime)')
        plt.grid(True)
        plt.show()

    def plot_memory_usage(self):
        plt.plot(self.memory_usage, color='orange', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Memory Usage (KB)')
        plt.title('Computational Efficiency (Memory Usage)')
        plt.grid(True)
        plt.show()

    def plot_constraint_violations(self):
        plt.plot(self.constraint_violations, color='red', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Total Constraint Violations')
        plt.title('Constraint Satisfaction (Feasibility)')
        plt.grid(True)
        plt.show()


# Run Genetic Algorithm
GA_Employees, GA_Tasks = DS.Generate_data(['A','B','C','D','E'],5,10)
GA = GeneticAlgorithm(GA_Employees, GA_Tasks, pop_size=20, generations=25, mutation_rate=0.2)
GA.evolve()

print(f"\nBest assignment (task → employee): {GA.best.assignment}")
print(f"Best cost: {GA.best.cost:.2f}")

# Plotting the results
GA.plot_cost()  # Solution Quality (Objective Function)
GA.plot_runtime()  # Computational Efficiency (Runtime)
GA.plot_memory_usage()  # Computational Efficiency (Memory Usage)
GA.plot_constraint_violations()  # Constraint Satisfaction (Feasibility)
