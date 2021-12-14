import ModalAnalysis as ma
import pandas as pd
import numpy as np


class MDBA:
    def __init__(self, n_vars,cost_func, population_size=40, Ub=0.99, Lb=0, num_iterations=100,A0=0.9,A_inf=0.6,r0=0.1,r_inf=0.7,fmin=0,fmax=1,guess='random'):
        self.population_size=population_size
        self.cost_func = cost_func

        self.ub=Ub
        self.lb=Lb
        self.n_vars=n_vars

        self.population, self.fitness = self.initialize_population(guess)

        
        self.loudness=A0*np.ones(shape=(self.population_size,1),dtype=np.float32)
        self.pulse_rate=r0*np.ones(shape=(self.population_size,1),dtype=np.float32)

        # algo parameters
        self.num_iterations = num_iterations
        self.frequency_range=np.array([fmin,fmax], dtype=np.float32)
        self.A0=A0
        self.A_inf=A_inf
        self.r0=r0
        self.r_inf=r_inf

        #results
        self.best_position = np.empty(shape=(1,n_vars),dtype=np.float32)
        self.best_fitness = np.inf
        self.best_bat = 0

    def initialize_population(self,guess):
        population = np.empty(shape=(self.population_size,self.n_vars),dtype=np.float32)
        fitness = np.empty(shape=(self.population_size,1),dtype=np.float32)

        if guess == 'random':
            population = self.lb+(self.ub-self.lb)*np.random.random(size=(self.population_size,self.n_vars))
        else:
            population = np.zeros(shape=(self.population_size,self.n_vars),dtype=np.float32)
        
        for i in range(self.population_size):
            fitness[i,0]=self.cost_func(population[i,:])
        
        return population, fitness
    
    def update_bat(self, bat, best, step):
        f1 = self.frequency_range[0]+(self.frequency_range[1]-self.frequency_range[0])*np.random.random((self.n_vars,))
        f2 = self.frequency_range[0]+(self.frequency_range[1]-self.frequency_range[0])*np.random.random((self.n_vars,))

        k=np.random.randint(0,self.population_size)
        while k==bat:
            k=np.random.randint(0,self.population_size)
        
        position=None
        if self.cost_func(self.population[k,:])<self.cost_func(self.population[bat,:]):
            position = self.population[bat,:]+(self.population[best,:]-self.population[bat,:])*f1+(self.population[k,:]-self.population[bat,:])*f2
        else:
            position = self.population[bat,:]+(self.population[best,:]-self.population[bat,:])*f1

        return position

    def run(self):
        best_fit=self.fitness.min()
        best_index=self.fitness.argmin()

        w_0=0.25*(self.ub-self.lb)
        w_inf=0.01*w_0
        log=np.zeros(shape=(self.num_iterations,))
        
        #iteration
        step=1
        while step <self.num_iterations+1:
            flag=False
            for bat in range(self.population_size):

                #update bat variables frequency, velocity, position
                position = self.update_bat(bat,best_index,step)
                position[position>self.ub]=self.ub
                position[position<self.lb]=self.lb
                
                #random walk: Local search stratergy
                if np.random.random()>self.pulse_rate[bat]:
                    w=(w_0-w_inf)*(step-self.num_iterations)/(1-self.num_iterations)+w_inf
                    position=position+self.loudness.mean()*np.random.uniform(-1,1,size=self.n_vars)*w
                
                    position[position>self.ub]=self.ub
                    position[position<self.lb]=self.lb


                #check bat fitness
                bat_fitness = self.cost_func(position)

                if np.random.random()<self.loudness[bat] and bat_fitness < self.fitness[bat,:]:
                    self.fitness[bat,:]=bat_fitness
                    self.population[bat,:]=position

                    self.pulse_rate[bat,:]=(self.r0-self.r_inf)*(step-self.num_iterations)/(1-self.num_iterations)+self.r_inf
                    self.loudness[bat,0]=(self.A0-self.A_inf)*(step-self.num_iterations)/(1-self.num_iterations)+self.A_inf
                    
                if bat_fitness < best_fit:
                    flag=True
                    best_fit = bat_fitness

                    self.fitness[best_index,:]=bat_fitness
                    self.population[best_index,:]=position

                    self.best_position = position.reshape(1,self.n_vars)
                    self.best_fitness = bat_fitness
            
            #elimination stratergy
            #eliminate poor performing individuals and randomly generate new solutions
            # if step%7==0:
            #     num=int(0.2*self.population_size)
            #     order=self.fitness.argsort(axis=0).reshape((self.population_size,))
                
            #     self.population=self.population[order,:]
            #     self.loudness=self.loudness[order]
            #     self.pulse_rate=self.pulse_rate[order]
            #     self.fitness=self.fitness[order]
                
            #     for i in range(1,num+1):
            #         self.population[-i,:self.n_vars//4]=self.population[np.random.randint(0,5),:self.n_vars//4]
            #         self.population[-i,self.n_vars//4:self.n_vars//2]=self.population[np.random.randint(0,5),self.n_vars//4:self.n_vars//2]
            #         self.population[-i,self.n_vars//2:3*self.n_vars//4]=self.population[np.random.randint(0,5),self.n_vars//2:3*self.n_vars//4]
            #         self.population[-i,3*self.n_vars//4:]=self.population[np.random.randint(0,5),3*self.n_vars//4:]
            #         self.fitness[-i,0]=self.cost_func(self.population[-i,:])
            
            if flag:
                print("{}: {}".format(step, best_fit))
                log[step-1]=self.best_fitness
                step+=1
        
        return log


        



if __name__=="__main__":
    def cost(x):
        return 20+np.e - 20*np.exp(-0.2*np.sqrt(np.sum(x*x)))-np.exp(np.sum(np.cos(2*np.pi*x)))
    
    optimizer = MDBA(n_vars=2,population_size=100,num_iterations=100,cost_func=cost,Ub=5,Lb=-5)
    
    optimizer.run()

    print(optimizer.best_position, optimizer.best_fitness)
    # print(cost(np.zeros(shape=(2,2))[1,:]))

