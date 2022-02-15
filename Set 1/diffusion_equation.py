import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

class Diffusion:
    def __init__(self, D=1, N=50):
        self.D = D
        self.N = N
        self.dx = self.dy = 1/N
        self.dt = self.dx**2/(4*D)

        self.t = 0

        self.c = np.zeros((N, N))
        # boundary conditions
        for i in range(N):
            self.c[0][i] = 1

        self.fig = plt.figure()

        self.running = True

    def analytic_sol(self, x, t, precision):
        if t == 0:
            return None

        c_sum = 0
        i = 0

        while True:
            to_add = math.erfc((1 - x + 2 * i) / (2 * (self.D * self.t)**0.5)) - math.erfc((1 + x + 2 * i) / (2 * (self.D * self.t)**0.5))
            c_sum += to_add
            i += 1

            if to_add < precision:
                break

        return c_sum

    def finite_diff(self):
        c_old = np.copy(self.c)
        self.t = round(self.t + self.dt, 8)
        print(self.t)

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

    def only_y(self):
        return np.array([np.mean(row) for row in self.c])

    def update(self):
        self.finite_diff()
        # self.jacobi_iter(10**-5)
        # self.gauss_seidel(10**-5)
        # self.sor(10**-8, 1.85)

        if not self.running:
            self.ani.event_source.stop()
            print("Stopping condition met!")

    def im_update(self, *args):
        self.update()
        self.im.set_array(self.c)

        return self.im,

    def line_update(self, *args):
        self.update()
        self.line1.set_ydata(self.only_y())
        self.line2.set_ydata([self.analytic_sol(x, self.t, 10**-5) for x in np.linspace(0, 1, len(self.c))])

        return self.line1, self.line2

    def im_animate(self):
        self.im = plt.imshow(self.c, cmap='bone', animated=True)
        self.ani = animation.FuncAnimation(self.fig, self.im_update, interval=0, blit=True)
        plt.show()

    def line_animate(self):
        self.line1, = plt.plot(np.linspace(0, 1, len(self.c)), self.only_y())
        self.line2, = plt.plot(np.linspace(1, 0, len(self.c)), [self.analytic_sol(x, self.t, 10**-4) for x in np.linspace(0, 1, len(self.c))])
        self.ani = animation.FuncAnimation(self.fig, self.line_update, interval=0, blit=True)
        plt.show()

    def ex_E(self):
        t_values = [round(0.1**i, 3) for i in np.linspace(0, 3, 7, endpoint = True)]
        differences = {}
        sim_data = {}

        while True:
            if self.t in t_values:
                analytic = np.array([self.analytic_sol(x, self.t, 10**-4) for x in np.linspace(1, 0, len(self.c))])
                simulate = self.only_y()

                differences[self.t] = simulate - analytic
                sim_data[self.t] = simulate

            if self.t == max(t_values):
                break

            self.update()
        
        for key, value in differences.items():
            plt.plot(np.linspace(0, 1, len(self.c)), value, label = key)

        plt.legend()
        plt.show()

        for key, value in sim_data.items():
            plt.plot(np.linspace(0, 1, len(self.c)), value, label = key)

        plt.legend()
        plt.show()

    def ex_F(self):
        t_values = [0, 0.001, 0.01, 0.1, 1]

        while True:
            if self.t in t_values:
                self.im = plt.imshow(self.c, cmap='bone', animated=True)
                plt.show()
            
            if self.t == max(t_values):
                break

            self.update()

    def ex_G(self):
        self.im_animate()


sim = Diffusion()
# sim.line_animate()
# sim.im_animate()
# sim.ex_E()
# sim.ex_F()
sim.ex_G()