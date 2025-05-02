import GeneticAlgorithm as GA           # Importing Genetic Algorithm module
import ParticleSwarmAlgorithm as PSO    # Importing Particle Swarm Optimization module
import AntColonyAlgorithm as ACO        # Importing Ant Colony Optimization module
import Data_Synthesizer as DS           # Importing Data Synthesizer module for synthetic data generation
import matplotlib.pyplot as plt         # Importing Matplotlib for plotting graphs

if __name__ == "__main__":              # Main execution block         
    # Importing the Data Synthesizer and generating synthetic data
    n_employees= 30 
    n_tasks = 50
    Employees, Tasks = DS.Generate_data(['A','B','C','D','E'], n_employees, n_tasks)  # Generate synthetic employee and task data

    # Running Genetic Algorithm (GA) for task assignment optimization
    GeneticAlgorithm = GA.GeneticAlgorithm(Employees, Tasks, pop_size=25, generations=500, mutation_rate=0.2)
    # Plot the performance evaluations of the Genetic Algorithm (GA)
    #GeneticAlgorithm.plot_cost()        # Plot all three performance evaluations: solution quality, memory usage, and constraint violations

    # Running Particle Swarm Optimization (PSO) for task assignment optimization
    PSOAlgorithm = PSO.Particle_Swarm_Optimiser(25, 0.3, 2, 3, Employees, Tasks, n_iter=500)
    # Plot the performance evaluations of the Particle Swarm Optimization (PSO)
    #PSOAlgorithm.plot_cost()            # Plot all three performance evaluations: solution quality, memory usage, and constraint violations

    # Running Ant Colony Optimization (ACO) for task assignment optimization
    ACOAlgorithm = ACO.AntColonyOptimser(25, 0.8, 0.02, Employees, Tasks, n_iter=500)
    # Plot the performance evaluations of the Ant Colony Optimization (ACO)
    #ACOAlgorithm.plot_cost()             # Plot all three performance evaluations: solution quality, memory usage, and constraint violations

    #comparison plotting between the algorithms 
    fig, axs = plt.subplots(2,2)
    plt.suptitle(f'Performance Graphs for {n_employees}-Employees and {n_tasks}=Tasks')
    axs[0,0].title.set_text("Cost over Iterations")
    axs[0,0].set_xlabel("Iterations")
    axs[0,0].set_ylabel("Cost")
    axs[0,0].plot(GeneticAlgorithm.best_costs,'b-',label='GA')
    axs[0,0].plot(PSOAlgorithm.gBestCostHistory,'r-',label='PSO')
    axs[0,0].plot(ACOAlgorithm.cost_history,'g-',label='ACO')
    axs[0,0].legend(loc='upper left')
    
    axs[0,1].title.set_text("Memory Used")
    axs[0,1].set_xlabel("Iterations")
    axs[0,1].set_ylabel("Cost")
    axs[0,1].plot(GeneticAlgorithm.memory_usage,'b-',label='GA')
    axs[0,1].plot(PSOAlgorithm.memoryuseHist,'r-',label='PSO')
    axs[0,1].plot(ACOAlgorithm.memoryuseHist,'g-',label='ACO')
    axs[0,1].legend(loc='upper left')
    
    axs[1,0].title.set_text("Feasibility")
    axs[1,0].set_xlabel("Iterations")
    axs[1,0].set_ylabel("# of Average Violations")
    axs[1,0].plot(GeneticAlgorithm.avg_constraint_violations,'b-',label='GA')
    axs[1,0].plot(PSOAlgorithm.averagetotalViolatioHist,'r-',label='PSO')
    axs[1,0].plot(ACOAlgorithm.averagetotalViolatioHist,'g-',label='ACO')
    axs[1,0].legend(loc='upper left')

    axs[1,1].title.set_text("Time Taken")
    axs[1,1].set_ylabel('Time taken to complete 500 iterations (seconds)')
    axs[1,1].bar(['GA','PSO','ACO'],[sum(GeneticAlgorithm.elapsed_times),sum(PSOAlgorithm.process_timeHist),sum(ACOAlgorithm.process_timeHist)])

    plt.show()
