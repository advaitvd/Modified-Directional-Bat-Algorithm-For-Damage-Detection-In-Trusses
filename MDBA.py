import ModalAnalysis as ma
import pandas as pd
import numpy as np


class MDBA:
    def __init__(self, n_vars,cost_func, population_size=10, Ub=1, Lb=0, num_iterations=100):
        self.population_size=population_size
        self.cost_func = cost_func

        self.ub=Ub
        self.lb=Lb
        self.n_vars=n_vars

        self.population, self.fitness = self.initialize_population()

        
        self.loudness=np.ones(shape=(self.population_size,1),dtype=np.float32)
        self.pulse_rate=np.zeros(shape=(self.population_size,1),dtype=np.float32)
        self.frequency=np.zeros(shape=(self.population_size,1),dtype=np.float32)
        self.velocity=np.zeros(shape=(self.population_size, n_vars),dtype=np.float32)

        # algo parameters
        self.num_iterations = num_iterations
        self.frequency_range=np.array([0,1], dtype=np.float32)
        self.loudness_decay = 0.9
        self.loudness_lim = 0.05
        self.pulse_rate_decay = 0.5
        self.gamma = 0.5

        #results
        self.best_position = np.empty(shape=(1,n_vars),dtype=np.float32)
        self.best_fitness = 0.0
        self.best_bat = 0

    def initialize_population(self):
        population = np.empty(shape=(self.population_size,self.n_vars),dtype=np.float)
        fitness = np.empty(shape=(self.population_size,1),dtype=np.float)

        population = self.lb+(self.ub-self.lb)*np.random.random(size=(self.population_size,self.n_vars))
        for i in range(self.population_size):
            fitness[i,0]=self.cost_func(population[i,:])
        
        return population, fitness
    
    def update_bat(self, bat, best):
        self.frequency[bat,:]=self.frequency_range[0]+(self.frequency_range[1]-self.frequency_range[0])*np.random.random()
        self.velocity[bat,:]=(self.population[best,:]-self.population[bat,:])*self.frequency[bat,:]
        position=self.population[bat,:]+self.velocity[bat,:]

        return position

    def run(self):
        best_fit=self.fitness.min()
        best_index=self.fitness.argmin()

        #iteration
        for step in range(1,self.num_iterations +1):
            for bat in range(self.population_size):
                
                #update bat variables
                position = self.update_bat(bat,best_index)

                #random walk
                if np.random.random()<self.loudness[bat,:]:
                    position+=self.loudness.mean()*np.random.uniform(-1,1,size=(self.n_vars,))
                
                #check limits
                position[position>self.ub]=self.ub
                position[position<self.lb]=self.lb

                #check bat fitness
                bat_fitness = self.cost_func(position)

                if bat_fitness < self.fitness[bat,:]:
                    self.fitness[bat,:]=bat_fitness
                    self.population[bat,:]=position
                    self.pulse_rate[bat,:]=self.pulse_rate_decay*(1-np.exp(-self.gamma*step))

                    if self.loudness[bat,0]>self.loudness_lim:
                        self.loudness[bat,0]*=self.loudness_decay
                    else:
                        self.loudness[bat,0]=self.loudness_lim
                    
                    if bat_fitness < best_fit:
                        best_fit = bat_fitness
                        best_index = bat

                        self.best_position = position.reshape(1,self.n_vars)
                        self.best_fitness = bat_fitness
                        self.best_bat = bat
        



if __name__=="__main__":
    # file_name = '2D-data.xlsx'
    # dimension = int(file_name[0])
    # elements = pd.read_excel(file_name, sheet_name='Sheet1').values[:, 1:]
    # nodes = pd.read_excel(file_name, sheet_name='Sheet2').values[:, 1:]
    # aa = ma.ModalAnalysis(elements, nodes, dimension)

    # M = aa.assembleMass()
    # K = aa.assembleStiffness(np.ones(elements.shape[0]))

    # w, v = aa.solve_eig(K, M)
    # print(w.shape, v.shape)
    # print(w)

    def cost(x):
        return 20+np.e - 20*np.exp(-0.2*np.sqrt(np.sum(x*x)))-np.exp(np.sum(np.cos(2*np.pi*x)))

    
    optimizer = MDBA(n_vars=2,population_size=100,num_iterations=100,cost_func=cost,Ub=5,Lb=-5)
    
    optimizer.run()

    print(optimizer.best_position, optimizer.best_fitness)
    # print(cost(np.zeros(shape=(2,2))[1,:]))

