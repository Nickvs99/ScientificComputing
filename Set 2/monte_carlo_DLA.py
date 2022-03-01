import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random

class MonteCarloDLA():

    """
    Implements Diffusion Limited Aggregation (DLA) through releasing
    random walkers.
    """

    def __init__(self, N=100):

        self.N = N
        self.grid = np.zeros((N, N))

        # Initialize structure at the centre bottom of the grid
        self.grid[0][N//2] = 1

        self.running = True

    def update(self):

        walker = RandomWalker(self.generate_random_walker_position(), bounds=[self.N, self.N])

        while True:
            
            walker.update()

            if not 0 < walker.position[1] < self.N - 1:
                
                # Respawn walker
                walker.position = self.generate_random_walker_position()

            if self.is_walker_connected(walker):
                
                # Add position of walker to the structure
                self.grid[walker.position[1]][walker.position[0]] = 1

                if walker.position[1] == self.N - 1:
                    self.running = False

                return

    def generate_random_walker_position(self):
        """ Generate a random position at the top. """

        return [random.randint(0, self.N - 1), self.N - 1]

    def is_walker_connected(self, walker):
        """ Checks if a random walker is connected to the structure. """

        for neighbour in walker.get_neighbours():
            neighbour_x, neighbour_y = neighbour
            if self.grid[neighbour_y][neighbour_x] == 1:
                return True

        return False

    def animate(self):

        fig = plt.figure()
        self.im = plt.imshow(self.grid, cmap='bone', origin='lower', extent=(0, 1, 0, 1), animated=True)

        self.ani = animation.FuncAnimation(fig, self.update_frame, interval=1, blit=True)
        plt.show()

    def update_frame(self, im_array):

        self.update()

        if not self.running:
            self.ani.event_source.stop()

        self.im.set_array(self.grid)
        return self.im,

class RandomWalker():

    def __init__(self, position, bounds):
        self.position = position
        self.bounds = bounds
    
    def update(self):
        
        # Get a random move: left, right, top, or bottom
        move = random.choice([[-1, 0], [1, 0], [0, 1], [0, -1]])

        for i in range(2):
            self.position[i] += move[i]

        # periodic boundaries in the x-direction
        self.position[0] = self.position[0] % self.bounds[0]

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

    sim = MonteCarloDLA(N=100)
    sim.animate()

    # TODO spawn walker only one row above the structure, instead of the top
    

if __name__ == "__main__":
    main()