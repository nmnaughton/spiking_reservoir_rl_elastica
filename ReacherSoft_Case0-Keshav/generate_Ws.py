import numpy as np
import scipy.sparse

def generate_Ws(seed=101, density=0.20, spectral_radius=0.9):

    bounds = [-1, 1]
    n_reservoir_neurons = 512
    input_size = 14  # 17 # 5
    output_size = 3  # 6  # 1

    np.random.seed(seed)
    W_in = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, input_size))
    # W_in /= input_size

    # Sample W_reservoir from a random unifrom distribution
    W_reservoir = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, n_reservoir_neurons))

    # Create a mask to make W_reservoir a sparse matrix with a density of 20%
    mask = scipy.sparse.rand(n_reservoir_neurons, n_reservoir_neurons, density=density)
    mask = np.array(mask.todense())
    mask[np.where(mask > 0)] = 1
    W_reservoir = W_reservoir * mask

    # Set the spectral radius of W_reservoir to 0.90
    E, _ = np.linalg.eig(W_reservoir)
    e_max = np.max(np.abs(E))
    W_reservoir /= np.abs(e_max)/spectral_radius
    # W_reservoir /= n_reservoir_neurons

    # Save W_in and W_reservoir to file
    np.save('W_in.npy', W_in)
    np.save('W_reservoir.npy', W_reservoir)

if __name__ == "__main__":
    generate_Ws()
