from diff_matrix import *
import scipy.linalg as la
import scipy.sparse.linalg as la_sparse
import matplotlib.pyplot as plt

class WaveEquation:
    def __init__(self, c=1, dx=0.01, L_x=1, L_y=1, circle=False, sparse=False):
        self.c = 1
        self.dx = dx
        self.L_x = L_x
        self.L_y = L_y
        self.sparse = sparse

        self.N = int(L_x / dx)
        self.M = int(L_y / dx)

        self.matrix = compute_matrix(self.M, self.N) / (dx ** 2)

        if circle:
            for j in range(self.M):
                for i in range(self.N):
                    if ((i + 0.5) - self.N / 2)**2/(self.N / 2)**2 + ((j + 0.5) - self.M / 2)**2/(self.M / 2)**2 > 1:
                        for m in range(self.N * self.M):
                            if m != j * self.N + i:
                                self.matrix[j * self.N + i][m] = 0

    def eigenvalues(self, k):
        if self.sparse:
            eigenvalues, eigenvectors = la_sparse.eigs(self.matrix, k=10, which='LR')
        else:
            eigenvalues, eigenvectors = la.eig(self.matrix)
        sort_indices = eigenvalues.argsort()[len(eigenvalues) - k:]

        return eigenvalues.real[sort_indices[::-1]], eigenvectors.T.real[sort_indices[::-1]]

    def show_eigenvectors(self, k=10):
        eigendata = self.eigenvalues(k)

        for i in range(len(eigendata[0])):
            eigenvalue = eigendata[0][i]
            eigenvector = eigendata[1][i]
            self.make_plot(np.reshape(eigenvector, (self.M, self.N)), (-eigenvalue)**0.5)

    def make_plot(self, v, title=None):
        plt.figure()
        im = plt.imshow(v, origin='lower', cmap='bone', extent=(0, self.L_x, 0, self.L_y))
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        if title:
            plt.title(title)
        plt.show()

if __name__ == "__main__":
    WaveEquation(dx=0.02, L_x = 1, circle=False).show_eigenvectors(10)
    WaveEquation(dx=0.02, L_x = 1, circle=True).show_eigenvectors(10)