from ModalAnalysis import ModalAnalysis as ma
from MDBA import MDBA as BatAlg
import numpy as np
import pandas as pd

def main():
    file_name = '2D-data.xlsx'
    dimension = int(file_name[0])
    elements = pd.read_excel(file_name, sheet_name='Sheet1').values[:, 1:]
    nodes = pd.read_excel(file_name, sheet_name='Sheet2').values[:, 1:]
    aa = ma(elements, nodes, dimension)
    M=aa.assembleMass()

    x_exp=np.zeros(len(elements))
    x_exp[5]=0.15
    x_exp[23]=0.1

    K=aa.assembleStiffness(x_exp)
    w_exp, v_exp=aa.solve_eig(K,aa.M)
    
    w_exp=w_exp[:5]
    v_exp=v_exp[:,:5]

    def objective_function(x):
        K=aa.assembleStiffness(x)
        w, v = aa.solve_eig(K, aa.M)
        w=w[:5]
        v=v[:,:5]
        MAC=None
        MACF=None
        MDLAC=None
        return np.sum(1-MAC)+np.sum(MDLAC)+np.sum(1-MACF)
    
    optimizer = BatAlg(n_vars=len(elements),population_size=10,num_iterations=10,cost_func=objective_function,Ub=1,Lb=0)
    
    optimizer.run()

    print(optimizer.best_position, optimizer.best_fitness)

if __name__=='__main__':
    main()