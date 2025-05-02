import GeneticAlgorithm as GA
import ParticleSwarmAlgorithm as PSO
import AntColonyAlgorithm as ACO
import Data_Synthesizer as DS
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Hello World!")

    # Importing the Data Synthesizer and generating synthetic data
    Employees, Tasks = DS.Generate_data(['A','B','C','D','E'], 5, 10)
    



    GeneticAlgorithm = GA.GeneticAlgorithm(Employees, Tasks, pop_size=20, generations=500, mutation_rate=0.2)
    GeneticAlgorithm.plot_cost()  # Plot all three performance evaluations


    PSOAlgorithm = PSO.Particle_Swarm_Optimiser(25,0.3,2,3,Employees,Tasks,n_iter=500)
    PSOAlgorithm.plot_cost()


    ACOAlgorithm = ACO.AntColonyOptimser(5,1,0.8,0.02,Employees,Tasks,patience=500)
    ACOAlgorithm.plot_cost()