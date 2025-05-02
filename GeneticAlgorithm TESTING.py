import numpy as np                  # Importing NumPy for numerical operations
import pandas as pd                 # Importing pandas for data manipulation (not used explicitly in this snippet)
import random as rd                 # Importing random for random number generation
import Data_Synthesizer as DS       # Importing custom module for data synthesis
import matplotlib.pyplot as plt     # Importing Matplotlib for plotting graphs
import copy                         # Importing copy for object copying (not used explicitly in this snippet)
import psutil                       # Importing psutil to track memory usage
import time                         # Importing time module to track elapsed time

# Penalty Weights (from Assignment Brief)
ALPHA = 0.20  # Overload penalty
BETA  = 0.20  # Skill mismatch penalty
DELTA = 0.20  # Difficulty violation penalty
GAMMA = 0.20  # Deadline violation penalty
SIGMA = 0.20  # Unique assignment penalty (not used in this encoding - applied directly in the cost function)

# GAIndividual: Represents one solution
class GAIndividual:
    def __init__(self, employees, tasks):
        # Initialize with employee and task data
        self.employees = employees                                                        # List of employees
        self.tasks = tasks                                                                # List of tasks
        self.assignment = [rd.randint(0, len(employees) - 1) for _ in range(len(tasks))]  # Randomly assign employees to tasks
        self.cost = self.calculate_cost()                                                 # Calculate the cost (penalty) for the initial solution

    def calculate_cost(self):
        # Method to calculate the cost of the solution (penalty for violations)
        employees = self.employees      # List of employees
        cost = 0                        # Initialize the cost variable

        # Reset task assignments for each employee
        for e in employees:
            e['Assigned Tasks'] = {}    # Clear previously assigned tasks

        # Assign tasks to employees
        for i, emp_id in enumerate(self.assignment):
            task_name = f"T{i}"                                             # Naming task as T followed by the index
            employees[emp_id]['Assigned Tasks'][task_name] = self.tasks[i]  # Assign the task to the employee

        # Loop through each employee to calculate penalties
        for emp in employees:
            assigned = list(emp['Assigned Tasks'].items())                           # Get all tasks assigned to the employee
            assigned.sort(key=lambda x: (x[1]['Deadline'], x[1]['Estimated Time']))  # Sort tasks by deadline and estimated time
            time_used = 0                                                            # Total time used by the employee
            penalty_skill = 0                                                        # Skill mismatch penalty
            penalty_difficulty = 0                                                   # Difficulty violation penalty
            penalty_deadline = 0                                                     # Deadline violation penalty

            # Calculate penalties for each task assigned to the employee
            for _, task in assigned:
                time_used += task['Estimated Time']                         # Add the estimated time for this task
                if task['Skills'] not in emp['Skills']:                     # Check if the employee has the required skill
                    penalty_skill += 1                                      # Add penalty for skill mismatch
                if task['Difficulty'] > emp['Skill_lvl']:                   # Check if the employee's skill level meets the task difficulty
                    penalty_difficulty += 1                                 # Add penalty for difficulty violation
                penalty_deadline += max(0, time_used - task['Deadline'])    # Calculate penalty if the deadline is missed

            penalty_overload = max(0, time_used - emp['Hours'])             # Add penalty for workload overload

            # Add up all penalties for the employee
            cost += (
                ALPHA * penalty_overload +
                BETA  * penalty_skill +
                DELTA * penalty_difficulty +
                GAMMA * penalty_deadline
            )

        return cost  # Return the total cost for the current assignment

    def mutate(self, mutation_rate=0.1):
        # Method to perform mutation on the solution (randomly reassign tasks to employees)
        for i in range(len(self.assignment)):
            if rd.random() < mutation_rate:                                  # If a random number is less than the mutation rate
                self.assignment[i] = rd.randint(0, len(self.employees) - 1)  # Reassign the task to a random employee

    def crossover(self, other):
        # Method to perform crossover between two solutions (parent individuals)
        point = rd.randint(1, len(self.assignment) - 1)                         # Random crossover point
        child1 = GAIndividual(self.employees, self.tasks)                       # Create a new child individual
        child2 = GAIndividual(self.employees, self.tasks)                       # Create another new child individual
        child1.assignment = self.assignment[:point] + other.assignment[point:]  # Combine part of both parents' assignments for child1
        child2.assignment = other.assignment[:point] + self.assignment[point:]  # Combine part of both parents' assignments for child2
        child1.cost = child1.calculate_cost()                                   # Calculate the cost for child1
        child2.cost = child2.calculate_cost()                                   # Calculate the cost for child2
        return child1, child2                                                   # Return both children

# Genetic Algorithm Optimiser
class GeneticAlgorithm:
    def __init__(self, employees, tasks, pop_size=10, generations=20, mutation_rate=0.1):
        # Initialize the genetic algorithm with employee data, task data, and GA parameters
        self.employees = employees                                                   # List of employees
        self.tasks = tasks                                                           # List of tasks
        self.population = [GAIndividual(employees, tasks) for _ in range(pop_size)]  # Create initial population
        self.generations = generations                                               # Number of generations to run the GA
        self.mutation_rate = mutation_rate                                           # Mutation rate
        self.best = min(self.population, key=lambda x: x.cost)                       # Set the best individual to the one with the lowest cost
        self.best_costs = []                                                         # List to store best costs over generations
        self.memory_usage = []                                                       # List to store memory usage per generation
        self.constraint_violations = []                                              # List to store constraint violations per generation
        self.avg_constraint_violations = []                                          # List to store average constraint violations per generation
        self.elapsed_times = []                                                      # List to store elapsed time per generation

        # Evolve the GA over generations to find the best solution
        for gen in range(self.generations):
            start_time = time.time()                                  # Start time for the current generation
            start_memory = psutil.Process().memory_info().rss / 1024  # Track memory usage at the start of the generation

            new_population = []                                       # List to store the new generation of solutions
            self.population.sort(key=lambda x: x.cost)                # Sort the population by cost (ascending)

            # Elitism: Carry over the best 2 individuals from the current population
            new_population.extend(self.population[:2])

            # Fill the rest of the population using crossover and mutation
            while len(new_population) < len(self.population):
                parent1 = self.tournament_selection()        # Select a parent using tournament selection
                parent2 = self.tournament_selection()        # Select another parent using tournament selection
                child1, child2 = parent1.crossover(parent2)  # Perform crossover to produce two children
                child1.mutate(self.mutation_rate)            # Mutate the first child
                child2.mutate(self.mutation_rate)            # Mutate the second child
                new_population.extend([child1, child2])      # Add the children to the new population

            self.population = new_population[:len(self.population)]    # Update the population with the new generation
            current_best = min(self.population, key=lambda x: x.cost)  # Find the best solution in the current generation
            if current_best.cost < self.best.cost:                     # If the current best solution has a lower cost
                self.best = current_best                               # Update the best solution

            # Track performance metrics for the current generation
            self.best_costs.append(self.best.cost)                            # Store the best cost
            current_memory_usage = psutil.Process().memory_info().rss / 1024  # Get the current memory usage
            self.memory_usage.append(current_memory_usage)                    # Store the memory usage

            # Track constraint violations for the current generation
            total_violations = 0
            for ind in self.population:
                total_violations += ind.cost                          # Sum up the penalties (cost) for all individuals

            avg_violations = total_violations / len(self.population)  # Calculate the average constraint violations
            self.avg_constraint_violations.append(avg_violations)

            # Track elapsed time for the current generation
            end_time = time.time()                   # End time for the current generation
            elapsed_time = end_time - start_time     # Calculate the time taken for the generation
            self.elapsed_times.append(elapsed_time)  # Store the elapsed time

            print(f"Generation {gen + 1} - Best Cost: {self.best.cost:.2f}")  # Print the best cost for the current generation

    def tournament_selection(self, k=3):
        # Tournament selection method to choose the best individual from a random sample of k individuals
        return min(rd.sample(self.population, k), key=lambda x: x.cost)  # Return the individual with the lowest cost

    def plot_cost(self):
        # Plotting the performance evaluations across generations
        plt.figure(figsize=(18, 6))                                    # Create a figure with specified size

        # Plot Solution Quality (Optimality) - Objective Function (Total Penalty)
        plt.subplot(1, 3, 1)                                           # Create subplot for the first plot
        plt.plot(self.best_costs, color='blue', linewidth=2)           # Plot the best costs across generations
        plt.xlabel('Generation')                                       # X-axis label
        plt.ylabel('Objective (Cost) Function Value (Total Penalty)')  # Y-axis label
        plt.title('Solution Quality (Optimality)')                     # Title of the plot
        plt.grid(True)                                                 # Add grid lines to the plot

        # Plot Computational Efficiency (Memory Usage)
        plt.subplot(1, 3, 2)                                           # Create subplot for the second plot
        plt.plot(self.memory_usage, color='orange', linewidth=2)       # Plot memory usage across generations
        plt.xlabel('Generation')                                       # X-axis label
        plt.ylabel('Memory Usage (KB)')                                # Y-axis label
        plt.title('Computational Efficiency (Memory Usage)')           # Title of the plot
        plt.grid(True)                                                 # Add grid lines to the plot

        # Plot Average Constraint Violations and Elapsed Time (with downsampling every 10th generation)
        plt.subplot(1, 3, 3)    # Create subplot for the third plot
        ax1 = plt.gca()         # Get the current axis for the plot

        # Downsample every 10th generation for constraint violations and elapsed time
        downsampled_gen = range(0, self.generations, 10)

        # Plot Average Constraint Violations (Primary Y-axis)
        ax1.plot(downsampled_gen, self.avg_constraint_violations[::10], color='red', linewidth=2)
        ax1.set_xlabel('Generation')                                          # X-axis label
        ax1.set_ylabel('Avg. Constraint Violations/Generation', color='red')  # Y-axis label
        ax1.tick_params(axis='y', labelcolor='red')                           # Set the color of the y-axis labels

        # Create a second Y-axis for Elapsed Time
        ax2 = ax1.twinx()                                                                # Create a twin axis to share the same X-axis
        ax2.plot(downsampled_gen, self.elapsed_times[::10], color='green', linewidth=2)  # Plot elapsed time
        ax2.set_ylabel('Elapsed Time/Generation (seconds)', color='green')               # Y-axis label for elapsed time
        ax2.tick_params(axis='y', labelcolor='green')                                    # Set the color of the y-axis labels

        # Title for Elapsed Time axis (Green)
        ax2.set_title('Average Constraint Violations and Elapsed Time Over Generations')
        plt.grid(True)                   # Add grid lines to the plot

        # Adjust spacing between subplots to avoid overlap
        plt.subplots_adjust(wspace=0.3)  # Increase space between subplots to avoid overlap

        # Ensure everything fits inside the plot
        plt.tight_layout()               # Make sure the layout does not overlap

        # Display the plots
        plt.show()

# # Used for testing the Genetic Algorithm
# # Importing the Data Synthesizer and generating synthetic data
# GA_Employees, GA_Tasks = DS.Generate_data(['A', 'B', 'C', 'D', 'E'], 5, 10)

# # Set up the GA for 500 generations
# GA = GeneticAlgorithm(GA_Employees, GA_Tasks, pop_size=20, generations=500, mutation_rate=0.2)
# GA.plot_cost()  # Plot all three performance evaluations