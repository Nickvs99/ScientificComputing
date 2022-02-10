import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Rope:

    def __init__(self, initial_position_table, c, dt, dx):

        self.initial_position_table = initial_position_table.copy()
        self.position_table = initial_position_table

        self.N = len(self.position_table[0])
        self.c = c
        self.dt = dt
        self.dx = dx

    def reset(self):
        self.position_table = self.initial_position_table

    def update(self):
        
        new_position = [self.calc_new_position_at_i(i) for i in range(self.N)]

        # Remove oldest entry from position table. They are no longer needed
        self.position_table.pop(0)

        self.position_table.append(new_position)

    def calc_new_position_at_i(self, i):
        """
        Calculates the new position of the rope at index i.

        Arguments:
            i (int): index of rope

        Returns:
            (float): new position of the rope at index i
        """
        # boundary conditions
        if i == 0 or i == self.N - 1:
            return 0

        # Finite difference equation for a 1d wave equation
        factor = (self.c * self.dt / self.dx) ** 2
        return factor * (self.position_table[1][i + 1] + self.position_table[1][i - 1] - 2 * self.position_table[1][i]) - self.position_table[0][i] + 2 * self.position_table[1][i]

    def animate(self):
        """ Show an animation of the rope. """

        fig, ax = plt.subplots()

        x = np.arange(0, 1 + self.dx, self.dx)
        self.line, = ax.plot(x, self.position_table[1])

        ani = animation.FuncAnimation(fig, self.update_frame, interval=1)
        
        plt.ylim(-1, 1)
        plt.show()

    def update_frame(self, i):
        """
        Update the frame for the animation

        Arguments:
            i (any): dummy variable needed for animation.FuncAnimation

        Returns:
            line: The updated values for the frame
        """ 

        self.update()
        self.line.set_ydata(self.position_table[1])
        return self.line,

    def plot(self, max_iterations=100, nr_plots=5):
        """
        Plot the current position of the rope after different elapsed time

        Arguments:
            max_iterations (int): The number of times the rope is updated.
            nr_plots (int): The number of times the current position is plotted. The rope is plotted
                after 'max_iterations // nr_plots' frames.
        """

        xvalues = np.arange(0, 1 + self.dx, self.dx)

        iterations_per_plot = max_iterations // nr_plots

        for i in range(max_iterations):
            
            self.update()
            if i % iterations_per_plot == 0:
                plt.plot(xvalues, self.position_table[1], label=f"t = {self.dt * i: .2f}")
        
        plt.legend()
        plt.show()

def main():

    dx = 0.01
    dt = 0.001

    initial_position_functions = (
        lambda x: math.sin(2 * math.pi * x),
        lambda x: math.sin(5 * math.pi * x),
        lambda x: math.sin(5 * math.pi * x) if 0.2 < x < 0.4 else 0,
    )

    for initial_position_function in initial_position_functions:

        initial_position = [initial_position_function(x) for x in np.arange(0, 1 + dx, dx)]

        initial_position_table = [initial_position, initial_position]  # No initial velocity

        rope = Rope(initial_position_table, 1, dt, dx)
        rope.plot()

        rope.reset()
        rope.animate()

if __name__ == "__main__":
    main()


