import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Diffusion:
    def __init__(self, D=1, N=50):
        self.D = D
        self.N = N
        self.dx = self.dy = 1/N
        self.dt = self.dx**2/(4*D)

        self.c = np.zeros((N, N))
        # boundary conditions
        for i in range(N):
            self.c[0][i] = 1

        self.fig = plt.figure()
        self.im = plt.imshow(self.c, cmap='bone', animated=True)

        self.running = True

    def finite_diff(self):
        c_old = self.c

        for i in range(self.N):
            for j in range(self.N):
                if 0 < j < self.N - 1:
                    self.c[j][i] = self.c[j][i] + (self.dt/(self.dx**2)) * self.D * (c_old[j][(i+1)%self.N] + c_old[j][i-1] + c_old[j+1][i] + c_old[j-1][i] - 4*c_old[j][i])

    def jacobi_iter(self, stopping_e):
        c_old = np.copy(self.c)

        for i in range(self.N):
            for j in range(self.N):
                if 0 < j < self.N - 1:
                    self.c[j][i] = 1/4 * (c_old[j][(i+1)%self.N] + c_old[j][i-1] + c_old[j+1][i] + c_old[j-1][i])

        if np.amax(np.absolute(self.c - c_old)) < stopping_e:
            self.running = False

    def gauss_seidel(self, stopping_e):
        self.sor(stopping_e, 1)

    def sor(self, stopping_e, omega):
        c_old = np.copy(self.c)

        for i in range(self.N):
            for j in range(self.N):
                if 0 < j < self.N - 1:
                    self.c[j][i] = omega/4 * (self.c[j][(i+1)%self.N] + self.c[j][i-1] + self.c[j+1][i] + self.c[j-1][i]) + (1 - omega) * c_old[j][i]

        if np.amax(np.absolute(self.c - c_old)) < stopping_e:
            self.running = False

    def update(self, *args):
        # self.finite_diff()
        # self.jacobi_iter(10**-5)
        # self.gauss_seidel(10**-5)
        self.sor(10**-8, 1.85)
        
        self.im.set_array(self.c)

        if not self.running:
            self.ani.event_source.stop()
            print("Stopping condition met!")

        return self.im,

    def animate(self):
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=0, blit=True, repeat=self.running)
        plt.show()

sim = Diffusion()
sim.animate()