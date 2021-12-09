import ModalAnalysis as ma
import pandas as pd
import numpy as np


def generate_solution(n):
    '''
    Generates a random solution for size n
    parameters:
        n - number of elements
    returns: randomly generated damage parameters vector of size n
    '''
    return 1-np.random.random(size=n)

def initialize_population(pop_size,n):
    '''
    Initializes population with size of pop_size and n number of elements.
    parameters: 
        pop_size - integer, population size
        n - integer, number of elements
    returns: list of numpy arrays of size pop_size where each array is of size n
    '''
    pop=[]
    for i in range(pop_size):
        pop.append(generate_solution(n))
    
    return pop

class MDBA:
    def __init__(self):
        pass


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

    print(initialize_population(10,5))