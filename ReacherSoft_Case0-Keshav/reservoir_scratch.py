import nengo
import nengo_loihi
import numpy as np
import scipy

if __name__ == "__main__":
    network = nengo.Network()
    W_in = np.load('W_in.npy')
    W_reservoir = np.load('W_reservoir.npy')

    with network:
        def get_scaled_rod_state(time):
            rod_state = np.random.rand(14)
            print(rod_state)
            return rod_state * 2000

        def set_scaled_reservoir_output(time, reservoir_output):
            reservoir_output = reservoir_output * 1e-4
            print(reservoir_output)
            return reservoir_output

        W_reservoir_sparse = scipy.sparse.find(W_reservoir)
        indicies = np.array([W_reservoir_sparse[0], W_reservoir_sparse[1]]).T
        W_reservoir_sparse_nengo = nengo.Sparse(W_reservoir.shape, indicies, W_reservoir_sparse[2])

        input_layer = nengo.Node(output=get_scaled_rod_state, size_in=0, size_out=14)
        reservoir = nengo.Ensemble(n_neurons=512, dimensions=14, neuron_type=nengo.LIF())
        process_node = nengo.Node(output=set_scaled_reservoir_output, size_in=512)

        conn_in = nengo.Connection(input_layer, reservoir.neurons, synapse=None, transform=W_in)
        conn_proc = nengo.Connection(reservoir.neurons, process_node, synapse=0.01)
        conn_res = nengo.Connection(reservoir.neurons, reservoir.neurons, transform=W_reservoir)

    sim = nengo_loihi.Simulator(network, progress_bar=False)
    sim.run(1)
