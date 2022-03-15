from diff_matrix import *
import scipy.linalg as la
from scipy.sparse import linalg as linalg2
import matplotlib.pyplot as plt

class WaveEquation:
    def __init__(self, c=1, dx=0.01, L_x=1, L_y=1, circle=False):
        self.c = 1
        self.dx = dx
        self.L_x = L_x
        self.L_y = L_y

        self.N = int(L_x / dx)
        self.M = int(L_y / dx)

        self.matrix = compute_matrix(self.M, self.N) / (dx ** 2)

    def eigenvalues(self):
        eigenvalues, eigenvectors = la.eig(self.matrix)
        sort_indices = eigenvalues.argsort()

        return eigenvalues.real[sort_indices[::-1]], eigenvectors.T.real[sort_indices[::-1]]

    def show_eigenvectors(self):
        for eigenvector in self.eigenvalues()[1]:
            self.make_plot(np.reshape(eigenvector.real, (self.M, self.N)))

    def make_plot(self, v):
        plt.figure()
        im = plt.imshow(v, origin='lower', cmap='bone', extent=(0, self.L_x, 0, self.L_y))
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

if __name__ == "__main__":
    WaveEquation(dx=0.04, L_x = 1).show_eigenvectors()