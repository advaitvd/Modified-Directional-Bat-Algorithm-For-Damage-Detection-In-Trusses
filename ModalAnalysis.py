import pandas as pd
import numpy as np
from scipy import linalg as LA


class ModalAnalysis:
    def __init__(self, elements, nodes, dimension):
        self.elements = elements
        self.nodes = nodes
        self.Ndfe = dimension*2
        self.Ne = elements.shape[0]
        self.lengths = self.findLengths(nodes, elements)
        self.densities = elements[:, 3]
        self.areas = elements[:, 4]
        self.Es = elements[:, 2]
        self.k = np.zeros((self.Ne, self.Ndfe, self.Ndfe))
        self.m = np.zeros((self.Ne, self.Ndfe, self.Ndfe))
        self.massMatrices()
        self.stiffnessMatrices()
        self.M=self.assembleMass()

    def findLengths(self, nodes, elements):
        lengths = np.zeros((self.Ne, 1))
        for i in range(self.Ne):
            lengths[i, 0] = np.linalg.norm(
                nodes[int(elements[i, 0])-1]-nodes[int(elements[i, 1])-1])

        return lengths

    def massMatrices(self):
        if self.Ndfe/2 == 3:
            arr = np.array([2, 0, 0, 1, 0, 0,
                            0, 2, 0, 0, 1, 0,
                            0, 0, 2, 0, 0, 1,
                            1, 0, 0, 2, 0, 0,
                            0, 1, 0, 0, 2, 0,
                            0, 0, 1, 0, 0, 2])
        elif self.Ndfe/2 == 2:
            arr = np.array([2, 0, 1, 0,
                            0, 2, 0, 1,
                            1, 0, 2, 0,
                            0, 1, 0, 2])

        for i in range(self.Ne):
            c = self.densities[i]*self.areas[i]*self.lengths[i]/6
            self.m[i, :, :] = c*(arr.reshape(self.Ndfe, self.Ndfe))

        return self.m

    def stiffnessMatrices(self):
        if self.Ndfe/2 == 3.0:
            for i in range(self.Ne):
                node1 = int(self.elements[i, 0])-1
                node2 = int(self.elements[i, 1])-1
                x1 = self.nodes[node1, 0]
                x2 = self.nodes[node2, 0]
                y1 = self.nodes[node1, 1]
                y2 = self.nodes[node2, 1]
                z1 = self.nodes[node1, 2]
                z2 = self.nodes[node2, 2]
                lamx = (x2-x1)/self.lengths[i, 0]
                lamy = (y2-y1)/self.lengths[i, 0]
                lamz = (z2-z1)/self.lengths[i, 0]
                k = np.array([lamx, lamy, lamz]).reshape(3, 1)
                k = k@k.transpose()
                k = np.append(k, -k, 1)
                k = np.append(k, -k, 0)
                self.k[i, :, :] = k * \
                    (self.Es[i]*self.areas[i]/self.lengths[i, 0])
        else:
            for i in range(self.Ne):
                node1 = int(self.elements[i, 0])-1
                node2 = int(self.elements[i, 1])-1
                x1 = self.nodes[node1, 0]
                x2 = self.nodes[node2, 0]
                y1 = self.nodes[node1, 1]
                y2 = self.nodes[node2, 1]
                lamx = (x2-x1)/self.lengths[i, 0]
                lamy = (y2-y1)/self.lengths[i, 0]
                k = np.array([lamx, lamy]).reshape(2, 1)
                k = k@k.transpose()
                k = np.append(k, -k, 1)
                k = np.append(k, -k, 0)
                self.k[i, :, :] = k * \
                    (self.Es[i]*self.areas[i]/self.lengths[i, 0])

        return self.k

    def assembleStiffness(self, x):
        n = self.nodes.shape[0]
        Ndf = int(self.Ndfe/2)
        K = np.zeros((n*Ndf, n*Ndf))

        for i in range(self.Ne):
            n1 = int(self.elements[i, 0])-1
            n2 = int(self.elements[i, 1])-1
            K[Ndf*n1:Ndf*(n1+1), Ndf*n1:Ndf*(n1+1)] += x[i] * \
                self.k[i, :Ndf, :Ndf]
            K[Ndf*n2:Ndf*(n2+1), Ndf*n2:Ndf*(n2+1)] += x[i] * \
                self.k[i, Ndf:, Ndf:]
            K[Ndf*n1:Ndf*(n1+1), Ndf*n2:Ndf*(n2+1)] += x[i] * \
                self.k[i, :Ndf, Ndf:]
            K[Ndf*n2:Ndf*(n2+1), Ndf*n1:Ndf*(n1+1)] += x[i] * \
                self.k[i, Ndf:, :Ndf]

        return K

    def assembleMass(self):
        n = self.nodes.shape[0]
        Ndf = int(self.Ndfe/2)
        M = np.zeros((n*Ndf, n*Ndf))

        for i in range(self.Ne):
            n1 = int(self.elements[i, 0])-1
            n2 = int(self.elements[i, 1])-1
            M[Ndf*n1:Ndf*(n1+1), Ndf*n1:Ndf*(n1+1)] += self.m[i, :Ndf, :Ndf]
            M[Ndf*n2:Ndf*(n2+1), Ndf*n2:Ndf*(n2+1)] += self.m[i, Ndf:, Ndf:]
            M[Ndf*n1:Ndf*(n1+1), Ndf*n2:Ndf*(n2+1)] += self.m[i, :Ndf, Ndf:]
            M[Ndf*n2:Ndf*(n2+1), Ndf*n1:Ndf*(n1+1)] += self.m[i, Ndf:, :Ndf]

        return M

    def solve_eig(self, K, M):
        w, v = LA.eig(np.matmul(np.linalg.inv(M), K))
        to_rm = []
        for i in range(w.shape[0]):
            if np.imag(w[i]) != 0.0 or np.real(w[i]) < 0.0 or np.abs(np.real(w[i])) < 1e-7:
                to_rm.append(i)

        w = np.real(np.delete(w, to_rm))
        v = np.real(np.delete(v, to_rm, 1))
        idx = np.arange(w.shape[0])
        zipped = zip(w, idx)
        zipped = sorted(zipped)
        w = np.array([i for i, _ in zipped])
        idx = [i for _, i in zipped]
        v[:, :] = v[:, idx]
        return (w, v)


if __name__ == "__main__":
    file_name = '2D-data.xlsx'
    dimension = int(file_name[0])
    elements = pd.read_excel(file_name, sheet_name='Sheet1').values[:, 1:]
    nodes = pd.read_excel(file_name, sheet_name='Sheet2').values[:, 1:]
    aa = ModalAnalysis(elements, nodes, dimension)
    M = aa.M
    K = aa.assembleStiffness(np.ones(elements.shape[0]))

    w, v = aa.solve_eig(K, M)
    print(w.shape, v.shape)
    print(w)
    # print(w[0])
    # print(v[:, 0])

    # A = np.matmul(np.linalg.inv(M), K)
    # B = w[0]*np.eye(K.shape[0])
    # print(np.matmul((A-B), v[:, 0]))
