import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random

class MonteCarloDLA():

    """
    Implements Diffusion Limited Aggregation (DLA) through releasing
    random walkers.
    """

    def __init__(self, N=100, sticking_probability=0.1):

        self.N = N
        self.grid = np.zeros((N, N))

        # Initialize structure at the centre bottom of the grid
        self.grid[0][N//2] = 1

        self.running = True

        # The heighest index of the structure
        self.structure_top_index = 0

        self.sticking_probability = sticking_probability

    def run(self):
        while self.running:
            self.update()

    def update(self):

        # The walker is initialized at one row above the structure. Its y-bounds is one extra layer tall,
        # otherwise it can get stuck when the structure is immediatly underneath it at initilization.
        walker = RandomWalker(self.generate_random_walker_position(), bounds=[self.N, self.structure_top_index + 2])

        while True:
            
            walker.update(self.grid)
                
            # Respawn walker when out of bounds
            if not 0 < walker.position[1] < self.N - 1:
                walker.position = self.generate_random_walker_position()

            if self.is_walker_connected(walker):
                
                if random.random() > self.sticking_probability:
                    continue

                # Add position of walker to the structure
                self.grid[walker.position[1]][walker.position[0]] = 1

                self.structure_top_index = max(self.structure_top_index, walker.position[1])
                
                # Simulation stops when structure has reached the top
                if walker.position[1] == self.N - 1:
                    self.running = False

                return

    def generate_random_walker_position(self):
        """ Generate a random position one row above the heighest cell of the structure. """

        return [random.randint(0, self.N - 1), self.structure_top_index + 1]

    def is_walker_connected(self, walker):
        """ Checks if a random walker is connected to the structure. """

        for neighbour in walker.get_neighbours():
            neighbour_x, neighbour_y = neighbour
            if self.grid[neighbour_y][neighbour_x] == 1:
                return True

        return False

    def animate(self):

        fig = plt.figure()
        self.im = plt.imshow(self.grid, cmap='bone', origin='lower', animated=True)
        self.ani = animation.FuncAnimation(fig, self.update_frame, interval=1, blit=True)
        plt.show()

    def update_frame(self, im_array):

        self.update()

        if not self.running:
            self.ani.event_source.stop()

        self.im.set_array(self.grid)
        return self.im,

    def create_plot(self):
        fig = plt.figure()
        plt.imshow(self.grid, cmap='bone', origin='lower')
        plt.title(f"p = {self.sticking_probability}")

class RandomWalker():

    def __init__(self, position, bounds):
        self.position = position
        self.bounds = bounds
    
    def update(self, grid):

        neighbours = self.get_neighbours()
        
        # Random walker can only move towards a cell which is not part of the structure
        while True:
            temp_position = random.choice(neighbours)
            
            if grid[temp_position[1]][temp_position[0]] == 0:
                break
            
        self.position = [temp_position[0], temp_position[1]]

    def get_neighbours(self):
        
        neighbours = []

        neighbour_offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for neighbour_offset in neighbour_offsets:
            
            # Calculate the neighbours position on the grid
            coordinate = [(self.position[0] + neighbour_offset[0]) % self.bounds[0], self.position[1] + neighbour_offset[1]]
            
            if self.is_in_bounds(coordinate):
                neighbours.append(coordinate)

        return neighbours

    def is_in_bounds(self, coordinate):

        return 0 <= coordinate[0] < self.bounds[0] and 0 <= coordinate[1] < self.bounds[1]


def main():

    ps = np.arange(0.1, 1.1, 0.1)
    for p in ps:
        print(f"\rRunning sticking probability: {p:.2f}", end="")
        
        sim = MonteCarloDLA(N=100, sticking_probability=p)
        sim.run()
        sim.create_plot()   
    
    plt.show()

if __name__ == "__main__":
    main()