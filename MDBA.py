import ModalAnalysis as ma
import pandas as pd
import numpy as np


class GenerateSol:
    def __init__(self,x,ma):
        K=ma.assembleStiffness(x)
        self.w,self.v=ma.solve_eig(K,ma.M)
        

class MDBA:
    def __init__(self):
        pass


if __name__=="__main__":
    file_name = '2D-data.xlsx'
    dimension = int(file_name[0])
    elements = pd.read_excel(file_name, sheet_name='Sheet1').values[:, 1:]
    nodes = pd.read_excel(file_name, sheet_name='Sheet2').values[:, 1:]
    aa = ma.ModalAnalysis(elements, nodes, dimension)

    M = aa.assembleMass()
    K = aa.assembleStiffness(np.ones(elements.shape[0]))

    w, v = aa.solve_eig(K, M)
    print(w.shape, v.shape)
    print(w)