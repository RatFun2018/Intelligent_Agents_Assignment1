import GeneticAlgorithm as GA           # Importing Genetic Algorithm module
import ParticleSwarmAlgorithm as PSO    # Importing Particle Swarm Optimization module
import AntColonyAlgorithm as ACO        # Importing Ant Colony Optimization module
import Data_Synthesizer as DS           # Importing Data Synthesizer module for synthetic data generation
import matplotlib.pyplot as plt         # Importing Matplotlib for plotting graphs

if __name__ == "__main__":              # Main execution block
    print("Program Initiated!")         # Validation message

    # Importing the Data Synthesizer and generating synthetic data
    Employees, Tasks = DS.Generate_data(['A','B','C','D','E'], 5, 10)  # Generate synthetic employee and task data

    # Running Genetic Algorithm (GA) for task assignment optimization
    GeneticAlgorithm = GA.GeneticAlgorithm(Employees, Tasks, pop_size=20, generations=500, mutation_rate=0.2)
    # Plot the performance evaluations of the Genetic Algorithm (GA)
    GeneticAlgorithm.plot_cost()        # Plot all three performance evaluations: solution quality, memory usage, and constraint violations

    # Running Particle Swarm Optimization (PSO) for task assignment optimization
    PSOAlgorithm = PSO.Particle_Swarm_Optimiser(25, 0.3, 2, 3, Employees, Tasks, n_iter=500)
    # Plot the performance evaluations of the Particle Swarm Optimization (PSO)
    PSOAlgorithm.plot_cost()            # Plot all three performance evaluations: solution quality, memory usage, and constraint violations

    # Running Ant Colony Optimization (ACO) for task assignment optimization
    ACOAlgorithm = ACO.AntColonyOptimser(25, 0.8, 0.02, Employees, Tasks, n_iter=500)
    # Plot the performance evaluations of the Ant Colony Optimization (ACO)
    ACOAlgorithm.plot_cost()             # Plot all three performance evaluations: solution quality, memory usage, and constraint violations

    #comparison plotting between the algorithms 
    fig, axs = plt.subplots(2,2)

    axs[0,0].plot(GeneticAlgorithm.best_costs,'b-',label='GA')
    axs[0,0].plot(PSOAlgorithm.gBestCostHistory,'r-',label='PSO')
    axs[0,0].plot(ACOAlgorithm.cost_history,'g-',label='ACO')
    axs[0,0].legend(loc='upper left')

    plt.show()
