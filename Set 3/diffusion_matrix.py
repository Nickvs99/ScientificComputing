import numpy as np

def main():

    n = 3
    m = 4

    print(compute_matrix(n, m))


def compute_matrix(n, m):
    
    matrix = np.zeros((n * m, n * m))

    for index in range(n * m):

        neighbour_indices = get_neighbours(index, n, m)
        
        matrix[index][index] = -len(neighbour_indices)

        for neighbour_indices in neighbour_indices:
            matrix[index][neighbour_indices] = 1

    return matrix

def get_neighbours(index, n, m):

    index_row = index // m
    index_column = index % m

    neighbour_indices = []  
    temp_neighbour_indices = [index - 1, index + 1, index - m, index + m]
    for neighbour_index in temp_neighbour_indices:


        if neighbour_index >= 0 and neighbour_index <= n * m - 1 and (neighbour_index  // m == index_row or neighbour_index % m == index_column):
            neighbour_indices.append(neighbour_index)

    return neighbour_indices
    

if __name__ == "__main__":
    main()