class params:
    #must be 3 for now. if increasing activation index needs to be updated for neuron params to be last 2 dimensions
    dimensions = 3
    
    neuron_count = 3900 #min 3900

    max_neurons_per_cluster = 3900

    activation_distance = 5

    neuon_db_path = "C:\\Users\\dmerg\\Documents\\Projects\\MDNN\\neurons.ann"

    n_outputs = 10

    annoy_build_trees = 100 #depricated

    min_input = -1

    max_input = 1

    neuron_min_vector = -10

    neuron_max_vector = 10

    bias_min = -10

    bias_max = 10

    supress_min = -0.01

    supress_max = 0.01

    n_neuron_shifts = 3 #depricated

    nlist = 100

