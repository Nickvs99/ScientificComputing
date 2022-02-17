import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

from shapes import Circle, Rectangle

class Diffusion:
    def __init__(self, D=1, N=50, objects=None):
        self.D = D
        self.N = N
        self.dx = self.dy = 1/N
        self.dt = self.dx**2/(4*D)

        self.t = 0

        self.c = np.zeros((N, N))
        # boundary conditions
        for i in range(N):
            self.c[0][i] = 1

        self.iterations = 0
        self.running = True

        if objects:
            self.sinks = self.determine_sink_points(objects)
        else:
            self.sinks = []

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

    def only_y(self):
        return np.array([np.mean(row) for row in self.c])
    
    def run_sor(self, omega):

        while self.running:
            self.sor(10**-5, omega)
            self.iterations += 1
    
    def im_update(self, *args):
        self.update()
        if not self.running:
            self.ani.event_source.stop()
        self.im.set_array(self.c)

        return self.im,

    def line_update(self, *args):
        self.update()
        self.line1.set_ydata(self.only_y())
        self.line2.set_ydata([self.analytic_sol(x, self.t, 10**-5) for x in np.linspace(0, 1, len(self.c))])

        return self.line1, self.line2

    def im_animate(self):
        fig = plt.figure()
        self.im = plt.imshow(self.c, cmap='gist_ncar', animated=True)
        self.ani = animation.FuncAnimation(fig, self.im_update, interval=0, blit=True)
        plt.show()

    def line_animate(self):
        fig = plt.figure()
        self.line1, = plt.plot(np.linspace(0, 1, len(self.c)), self.only_y())
        self.line2, = plt.plot(np.linspace(1, 0, len(self.c)), [self.analytic_sol(x, self.t, 10**-4) for x in np.linspace(0, 1, len(self.c))])
        self.ani = animation.FuncAnimation(fig, self.line_update, interval=0, blit=True)
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

    def ex_H(self):
        while self.running:
            self.update()

    def ex_I(self):
        deltas = []

        while self.running:
            self.update()
            deltas.append(self.delta)

        plt.plot(deltas)
        plt.show()


    def determine_sink_points(self, objects):
        """
        Returns which indices become sink points

        Arguments:
            objects: a list of Shape

        Returns:
            sinks: set of index coordinates
                example structure: set( (i1, j1), (i2, j2), ...)
        """

        sinks = set()

        if objects:
            for obj in objects:
                
                for i in range(self.N):
                    for j in range(self.N):
                        
                        coordinate = (i / self.N, j / self.N)
                        index_coordinate = (i, j)
                        if index_coordinate in sinks:
                            continue

                        if obj.contains(coordinate):
                            sinks.add(index_coordinate)

        return sinks

class FiniteDifference(Diffusion):
    def __init__(self, D=1, N=50, objects=None):
        super().__init__(D, N, objects)

    def __str__(self):
        return "Finite difference"

    def update(self):
        c_old = np.copy(self.c)
        self.t = round(self.t + self.dt, 8)
        print(self.t)

        for i in range(self.N):
            for j in range(self.N):

                if (i, j) in self.sinks:
                    continue

                if 0 < j < self.N - 1:
                    self.c[j][i] = self.c[j][i] + (self.dt/(self.dx**2)) * self.D * (c_old[j][(i+1)%self.N] + c_old[j][i-1] + c_old[j+1][i] + c_old[j-1][i] - 4*c_old[j][i])

class JacobiIteration(Diffusion):
    def __init__(self, D=1, N=50, stopping_e=10**-5, objects=None):
        super().__init__(D, N, objects)

        self.stopping_e = stopping_e

    def __str__(self):
        return "Jacobi"

    def update(self):
        c_old = np.copy(self.c)

        for i in range(self.N):
            for j in range(self.N):

                if (i, j) in self.sinks:
                    continue

                if 0 < j < self.N - 1:
                    self.c[j][i] = 1/4 * (c_old[j][(i+1)%self.N] + c_old[j][i-1] + c_old[j+1][i] + c_old[j-1][i])

        self.delta = np.amax(np.absolute(self.c - c_old))

        if self.delta < self.stopping_e:
            self.running = False
            print("Stopping condition met!")

class GaussSeidel(Diffusion):
    def __init__(self, D=1, N=50, stopping_e=10**-5, objects=None):
        super().__init__(D, N, objects)

        self.stopping_e = stopping_e

    def __str__(self):
        return "Gauss-Seidel"

    def update(self):
        c_old = np.copy(self.c)

        for i in range(self.N):
            for j in range(self.N):

                if (i, j) in self.sinks:
                    continue

                if 0 < j < self.N - 1:
                    self.c[j][i] = 1/4 * (self.c[j][(i+1)%self.N] + self.c[j][i-1] + self.c[j+1][i] + self.c[j-1][i])
       
        self.delta = np.amax(np.absolute(self.c - c_old))

        if self.delta < self.stopping_e:
            self.running = False
            print("Stopping condition met!")

class SuccessiveOverRelaxation(Diffusion):
    def __init__(self, D=1, N=50, stopping_e=10**-5, omega=1.85, objects=None):
        super().__init__(D, N, objects)

        self.stopping_e = stopping_e
        self.omega = omega

    def __str__(self):
        return "SOR"

    def update(self):
        c_old = np.copy(self.c)

        for i in range(self.N):
            for j in range(self.N):

                if (i, j) in self.sinks:
                    continue

                if 0 < j < self.N - 1:
                    self.c[j][i] = self.omega/4 * (self.c[j][(i+1)%self.N] + self.c[j][i-1] + self.c[j+1][i] + self.c[j-1][i]) + (1 - self.omega) * c_old[j][i]
       
        self.delta = np.amax(np.absolute(self.c - c_old))

        if self.delta < self.stopping_e:
            self.running = False
            print("Stopping condition met!")
            
def iterations_needed(sim, omega):

    copy_sim = copy.deepcopy(sim)
    copy_sim.run_sor(omega)
    return copy_sim.iterations

def calc_optimal_omega(sim, a=1.7, b=2, tolerance=0.01):
    """
    Finds the omega value for which the simulations converges the fastest. It does this by
    performing Golden Section Search.

    Arguments:
        sim (Diffusion): the diffusion object
        a (float): minimum omega value
        b (float): maximum omega value
        tolerance (float): precision of the calculation

    Returns:
        omega (float)
    """

    t = (5 ** 0.5 - 1) / 2

    x1 = a + (1 - t) * (b - a)
    f1 = iterations_needed(sim, x1)
    x2 = a + t * (b - a)
    f2 = iterations_needed(sim, x2)

    while (b - a) > tolerance:

        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + t * (b - a)
            f2 = iterations_needed(sim, x2)

        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - t) * (b - a)
            f1 = iterations_needed(sim, x1)

    return (x1, f1) if f1 < f2 else (x2, f2)

def ex_K():

    N_values = np.linspace(5, 200, 15, dtype=int)
    
    omegas_no_objects = []
    omegas_with_objects = []

    iterations_no_objects = []
    iterations_with_objects = []

    for N in N_values:
        print(f"\rCalculating optimal omega for N={N}", end="")

        sim = Diffusion(N=N)
        optimal_omega, optimal_iteration = calc_optimal_omega(sim)
        omegas_no_objects.append(optimal_omega)
        iterations_no_objects.append(optimal_iteration)

        objects = [
            Rectangle((0.2, 0.2), 0.2, 0.05),
            Rectangle((0.4, 0.4), 0.2, 0.05),
            Circle((0.8, 0.2), 0.1),
        ]

        sim = Diffusion(objects=objects, N=N)
        optimal_omega, optimal_iteration = calc_optimal_omega(sim)
        omegas_with_objects.append(optimal_omega)
        iterations_with_objects.append(optimal_iteration)

    plt.plot(N_values, omegas_no_objects, label="No objects")
    plt.plot(N_values, omegas_with_objects, label="With objects")

    plt.xlabel("N")
    plt.ylabel("Optimal omega")
    plt.legend()
    plt.show()

    plt.plot(N_values, iterations_no_objects, label="No objects")
    plt.plot(N_values, iterations_with_objects, label="With objects")

    plt.xlabel("N")
    plt.ylabel("Iterations")
    plt.legend()
    plt.show()  

objects = [
    Rectangle((0.2, 0.2), 0.1, 0.05),
    Rectangle((0.4, 0.4), 0.1, 0.05),
    Circle((0.8, 0.2), 0.05)
]

# sim = SuccessiveOverRelaxation(objects=objects)
# sim = SuccessiveOverRelaxation()
# sim.line_animate()
# sim.im_animate()
# sim.ex_E()
# sim.ex_F()
# sim.ex_G()
# sim.ex_I()

# exercise I
for sim in [JacobiIteration(), GaussSeidel(), SuccessiveOverRelaxation()]:
    deltas = []

    while sim.running:
        sim.update()
        deltas.append(sim.delta)

    plt.plot(deltas, label=str(sim))

plt.xlim(-2, 22)
plt.legend()
plt.show()

ex_K()
