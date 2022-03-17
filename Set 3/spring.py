import math
import matplotlib.pyplot as plt
import numpy as np


class Spring:

    def __init__(self, initial_position, initial_velocity, m=1, k=1, dt=0.01, driving_force_amplitude=0, omega=1):

        self.x_values = [initial_position]
        self.v_values = [initial_velocity]
        self.t_values = [0]

        self.m = m
        self.k = k
        self.dt = dt

        # Driving force parameters
        self.A = driving_force_amplitude
        self.omega = omega

        self.force = [self.calc_sinus_force()]

    def run(self, iterations=100):

        for i in range(iterations):
            self.update()

    def update(self):
        
        self.x_values.append( self.x_values[-1] + self.v_values[-1] * self.dt)
        
        self.force.append(self.calc_sinus_force())
        self.v_values.append( self.v_values[-1] + (self.calc_hookes_force() + self.calc_sinus_force())/ self.m * self.dt)
        
        self.t_values.append(self.t_values[-1] + self.dt)

    def calc_hookes_force(self):

        return - self.k * self.x_values[-1]

    def calc_sinus_force(self):

        return self.A * math.sin(self.omega * self.t_values[-1])

    def create_plot(self):

        plt.figure()
        plt.plot(self.t_values, self.x_values, label="position")
        plt.plot(self.t_values, self.v_values, label="velocity")
        # plt.plot(self.t_values, self.force, label="force")

        plt.xlabel("t")

        plt.legend()

    def create_phase_plot(self):

        plt.figure()
        plt.plot(self.v_values, self.x_values)
        plt.xlabel("v")
        plt.ylabel("x")

def main():

    # for k in np.logspace(-1, 1, 10):
    #     spring = Spring(0.5, 0, dt=0.01, k=k, driving_force_amplitude=1)
    #     spring.run(iterations=10000)
    #     spring.create_plot()

    # plt.show()

    for omega in np.logspace(-1, 1, 10):

        spring = Spring(0.5, 0, driving_force_amplitude=1, omega=omega)
        spring.run(iterations=10000)
        spring.create_phase_plot()

    plt.show()

if __name__ == "__main__":
    main()