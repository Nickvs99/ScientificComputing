from diff_matrix import *
import scipy.linalg as la
import scipy.sparse.linalg as la_sparse
import matplotlib.pyplot as plt
import time

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
        start = time.time()

        if self.sparse:
            eigenvalues, eigenvectors = la_sparse.eigs(self.matrix, k=k, which='LR')
        else:
            eigenvalues, eigenvectors = la.eig(self.matrix)

        duration = time.time() - start
        print(f"{duration} s")
        
        sort_indices = eigenvalues.argsort()[len(eigenvalues) - k:]

        return eigenvalues.real[sort_indices[::-1]], eigenvectors.T.real[sort_indices[::-1]], duration

    def time_dependence(t, eigenvalue, A=1, B=1):
        eigenfreq = (-eigenvalue)**0.5
        return A * np.cos(self.c * eigenfreq * t) + B * np.sin(self.c * eigenfreq * t)

    def show_eigenvectors(self, k=10):
        eigendata = self.eigenvalues(k)

        for i in range(len(eigendata[0])):
            eigenvalue = eigendata[0][i]
            eigenvector = eigendata[1][i]
            self.make_plot(np.reshape(eigenvector, (self.M, self.N)), (-eigenvalue)**0.5)

    def im_animate(self, cmap='gist_ncar'):
        fig = plt.figure()
        self.im = plt.imshow(self.c, cmap=cmap, origin='lower', extent=(0, 1, 0, 1), animated=True)
        self.text = plt.text(.5, 2, '')
        self.ani = animation.FuncAnimation(fig, self.im_update, interval=1, blit=True)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def make_plot(self, v, title=None):
        plt.figure()
        im = plt.imshow(v, origin='lower', cmap='bone', extent=(0, self.L_x, 0, self.L_y))
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        if title:
            plt.title(title)
        plt.show()

# class Eigenmode:
#     def __init__(self, eigenvector, eigenvalue):


if __name__ == "__main__":
    WaveEquation(dx=0.01, L_x = 1, circle=False, sparse=True).show_eigenvectors(10)
    WaveEquation(dx=0.01, L_x = 1, circle=False, sparse=False).show_eigenvectors(10)
    # WaveEquation(dx=0.02, L_x = 1, circle=True).show_eigenvectors(10)