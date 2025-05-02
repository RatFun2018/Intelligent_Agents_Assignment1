import GeneticAlgorithm as GA
import ParticleSwarmAlgorithm as PSO
import AntColonyAlgorithm as ACO
import Data_Synthesizer as DS
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Hello World!")

    # Importing the Data Synthesizer and generating synthetic data
    Employees, Tasks = DS.Generate_data(['A','B','C','D','E'], 5, 10)
    


    # Set up the GA for 500 generations
    GeneticAlgorithm = GA.GeneticAlgorithm(Employees, Tasks, pop_size=20, generations=500, mutation_rate=0.2)
    # Evolve the GA to calculate results
    GeneticAlgorithm.evolve()

    # Now the GA is evolved, so you can safely print the best assignment and best cost
    print(f"\nBest assignment (task → employee): {GeneticAlgorithm.best.assignment}")
    print(f"Best cost: {GeneticAlgorithm.best.cost:.2f}")

    # Plotting the results
    plt.figure(figsize=(18, 6))  # Increase figure size to fit all 3 plots side by side
    GeneticAlgorithm.plot_cost()  # Solution Quality (Objective Function)
    GeneticAlgorithm.plot_memory_usage()  # Computational Efficiency (Memory Usage)
    GeneticAlgorithm.plot_constraint_violations()  # Constraint Satisfaction (Feasibility) and Elapsed Time

    # Adjusting spacing between subplots
    plt.subplots_adjust(wspace=0.3)  # Increase space between subplots to avoid overlap

    plt.tight_layout()  # Ensures everything fits inside the plot
    plt.show()



    PSOAlgorithm = PSO.Particle_Swarm_Optimiser(25,0.3,2,3,Employees,Tasks,n_iter=500)
    PSOAlgorithm.plot_cost()


    ACOAlgorithm = ACO.AntColonyOptimser(5,1,0.8,0.02,Employees,Tasks,patience=500)
    ACOAlgorithm.plot_cost()