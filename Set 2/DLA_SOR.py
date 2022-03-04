import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

# from shapes import Circle, Rectangle

class DLA_SOR:
    def __init__(self, stopping_e=10**-3, omega=1.9, eta=100, D=1, N=100, object_points=None):
        self.D = D
        self.N = N
        self.dx = self.dy = 1/N
        self.dt = self.dx**2/(4*D)

        self.stopping_e = stopping_e
        self.omega = omega
        self.eta = eta

        self.delta = float('inf')

        self.t = 0

        self.c = np.zeros((N, N))
        # boundary conditions
        for i in range(N):
            self.c[N-1][i] = 1
            for j in range(N):
                self.c[j][i] = j/self.N

        self.iterations = 0
        self.sor_iterations = 0
        self.running = True

        self.object_points = object_points
        if self.object_points == None:
            self.object_points = set([(int(N/2), 0)])
    
    def __str__(self):

        return f"DLA_SOR: N={self.N}, eta={self.eta}, omega={self.omega}, stopping_e={self.stopping_e}"

    def analytic_sol(self, x, t, precision):
        if t == 0:
            return None

        c_sum = 0
        i = 0

        while True:
            to_add = math.erfc((1 - x + 2 * i) / (2 * (self.D * t)**0.5)) - math.erfc((1 + x + 2 * i) / (2 * (self.D * t)**0.5))
            c_sum += to_add
            i += 1

            if to_add < precision:
                break

        return c_sum

    def solve(self):
        while self.delta > self.stopping_e:
            c_old = np.copy(self.c)

            for i in range(self.N):
                for j in range(self.N):

                    if (i, j) in self.object_points:
                        self.c[j][i] = 0
                        continue

                    if 0 < j < self.N - 1:
                        self.c[j][i] = self.omega/4 * (self.c[j][(i+1)%self.N] + self.c[j][i-1] + self.c[j+1][i] + self.c[j-1][i]) + (1 - self.omega) * c_old[j][i]
        
            self.delta = np.amax(np.absolute(self.c - c_old))

            self.sor_iterations += 1

        self.delta = float('inf')

    def grow(self):
        growth_candidates = []

        for point in self.object_points:
            for neighbor in ((point[0]+1,point[1]),(point[0]-1,point[1]),(point[0],point[1]+1),(point[0],point[1]-1)):
                if 0 <= neighbor[0] < self.N and 0 <= neighbor[1] < self.N and neighbor not in self.object_points and neighbor not in growth_candidates:
                    growth_candidates.append(neighbor)

        grow_probs = []

        for candidate in growth_candidates:
            grow_probs.append(abs(self.c[candidate[1]][candidate[0]])**self.eta)
    
        prob_sum = sum(grow_probs)
        grow_probs = [prob/prob_sum for prob in grow_probs]
        grow_choice = np.random.choice(len(growth_candidates), p=grow_probs)
        self.object_points.add(growth_candidates[grow_choice])
        
        # Simulation stops when structure has reached the top
        if growth_candidates[grow_choice][1] == self.N - 1:
             self.running = False

    def update(self):
        self.solve()
        self.grow()

        if self.delta < self.stopping_e:
            self.running = False
            print("Stopping condition met!")

    def only_y(self):
        return np.array([np.mean(row) for row in self.c])
    
    def run(self):
        while self.running:
            self.update()
            self.iterations += 1
        
    def calc_density(self):

        return len(self.object_points) / (self.N * self.N)
    
    def im_update(self, *args):
        self.update()

        if not self.running:
            self.ani.event_source.stop()

        sink_c = np.copy(self.c)

        for i in range(self.N):
            for j in range(self.N):
                if (i, j) in self.object_points:
                    sink_c[j][i] = 1
            
        self.im.set_array(sink_c)

        if self.t > 0:
            self.text.set_text(f"t = {self.t}")

        return self.im, self.text,

    def line_update(self, *args):
        self.update()
        self.line1.set_ydata(self.only_y())
        self.line2.set_ydata([self.analytic_sol(x, self.t, 10**-5) for x in np.linspace(0, 1, len(self.c))])

        return self.line1, self.line2

    def im_animate(self, cmap='gist_ncar'):
        fig = plt.figure()

        self.im = plt.imshow(self.c, cmap=cmap, origin='lower', extent=(0, 1, 0, 1), animated=True)
        self.text = plt.text(.5, 2, '')
        self.ani = animation.FuncAnimation(fig, self.im_update, interval=1, blit=True)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def line_animate(self):
        fig = plt.figure()
        self.line1, = plt.plot(np.linspace(0, 1, len(self.c)), self.only_y())
        self.line2, = plt.plot(np.linspace(0, 1, len(self.c)), [self.analytic_sol(x, self.t, 10**-4) for x in np.linspace(0, 1, len(self.c))])
        self.ani = animation.FuncAnimation(fig, self.line_update, interval=1, blit=True)
        plt.show()

    def create_plot(self, cmap='gist_ncar'):

        fig = plt.figure()
        
        sink_c = np.copy(self.c)

        for i in range(self.N):
            for j in range(self.N):
                if (i, j) in self.object_points:
                    sink_c[j][i] = 1

        plt.imshow(sink_c, cmap=cmap, origin='lower', extent=(0, 1, 0, 1))

        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")

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

if __name__ == "__main__":
    test = DLA_SOR()
    test.im_animate()
            
def iterations_needed(sim, omega):

    copy_sim = copy.deepcopy(sim)
    copy_sim.omega = omega
    copy_sim.run()
    return copy_sim.sor_iterations

def calc_optimal_omega(sim, a=1, b=2, tolerance=0.01):
    """
    Finds the omega value for which the simulations converges the fastest. It does this by
    performing Golden Section Search.

    Arguments:
        sim (Diffusion): the diffusion object
        a (float): minimum omega value
        b (float): maximum omega value
        tolerance (float): precision of the calculation

    Returns:
        tuple: (omega (float), iterations (int))
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