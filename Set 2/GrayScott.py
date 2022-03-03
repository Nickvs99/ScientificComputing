import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from matplotlib.colors import Normalize

class GrayScott:
    def __init__(self, N=100, dt=1, dx=1, Du=0.16, Dv=0.08, f=0.035, k=0.06, square_size = 0.5, noise=0):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.f = f
        self.k = k
        self.Du = Du
        self.Dv = Dv

        self.t = 0

        # initial conditions
        self.u = np.full((N, N), 0.5)
        self.v = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if (0.5 - square_size/2)*N < i < (0.5 + square_size/2)*N and (0.5 - square_size/2)*N < j < (0.5 + square_size/2)*N:
                    self.v[j, i] = 0.25

                self.u[j,i] = min(0.95, max(0, self.u[j,i] + np.random.normal(0, noise)))
                self.v[j,i] = min(0.95, max(0, self.v[j,i] + np.random.normal(0, noise)))

        self.iterations = 0
        self.running = True
    
    def run(self):
        while self.running:
            self.update()
            self.iterations += 1
    
    def im_update(self, *args):
        self.update()

        if not self.running:
            self.ani.event_source.stop()
            
        self.im_u.set_array(self.u)
        self.im_v.set_array(self.v)

        # if self.t > 0:
        #     self.text.set_text(f"t = {self.t}")

        return self.im_u, self.im_v, #self.text,

    def im_animate(self, cmap='gist_ncar'):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,6))
        self.im_u = ax1.imshow(self.u, cmap=cmap, norm=Normalize(0, 1), origin='lower', extent=(0, round(self.dx*self.N,8), 0, round(self.dx*self.N,8)), animated=True)
        self.im_v = ax2.imshow(self.v, cmap=cmap, norm=Normalize(0, 1), origin='lower', extent=(0, round(self.dx*self.N,8), 0, round(self.dx*self.N,8)), animated=True)
        ax1.set_title("u", fontsize=20)
        ax2.set_title("v", fontsize=20)
        # self.text = ax1.text(2, 2, 'gfgs')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig.colorbar(self.im_u, ax=ax1, fraction=0.046, pad=0.04)
        fig.colorbar(self.im_v, ax=ax2, fraction=0.046, pad=0.04)
        
        self.ani = animation.FuncAnimation(fig, self.im_update, interval=1, blit=True)
        plt.show()

    def update(self):
        u_old = np.copy(self.u)
        v_old = np.copy(self.v)
        self.t = round(self.t + self.dt, 8)

        for i in range(self.N):
            for j in range(self.N):
                self.u[j][i] += self.dt * (self.Du * (u_old[j][(i+1)%self.N] + u_old[j][i-1] + u_old[(j+1)%self.N][i] + u_old[j-1][i] - 4*u_old[j][i])/(self.dx**2) 
                                           - u_old[j][i] * v_old[j][i]**2 + self.f * (1 - u_old[j][i]))

        for i in range(self.N):
            for j in range(self.N):
                self.v[j][i] += self.dt * (self.Dv * (v_old[j][(i+1)%self.N] + v_old[j][i-1] + v_old[(j+1)%self.N][i] + v_old[j-1][i] - 4*v_old[j][i])/(self.dx**2) 
                                           + u_old[j][i] * v_old[j][i]**2 - (self.f + self.k) * v_old[j][i])

GrayScott(noise=0).im_animate() #'nipy_spectral')