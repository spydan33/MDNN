import numpy as np
import random
import sqlite3
from params import params
from annoy import AnnoyIndex
class MDNN:
    def build(n_inputs,n_neurons,child_id = 1):
        index = AnnoyIndex(params.dimensions + 2, 'euclidean')
        for i in range(n_inputs):
            #minimum vector value for input_neurons
            min_value = params.min_input
            #minimum vector value for input_neurons
            max_value = params.max_input
            #bias required to keep shape of vector
            bias = 0
            #supress also required to keep shape
            supress = 1
            input_neuron = np.random.uniform(min_value, max_value, size= params.dimensions)
            input_neuron = np.append(input_neuron,bias)
            input_neuron = np.append(input_neuron,supress)
            index.add_item(i, input_neuron)
        for i in range(n_neurons):
            #minimum vector value for neurons
            min_value = params.neuron_min_vector
            #minimum vector value for neurons
            max_value = params.neuron_max_vector
            #generate bias between ranges
            bias = random.uniform(params.bias_min,params.bias_max)
            #generate bias between ranges
            supress = random.uniform(params.supress_min,params.supress_max)
            #generate neurons randomly in 3d
            neuron = np.random.uniform(min_value, max_value, size= params.dimensions)
            neuron = np.append(neuron,bias)
            neuron = np.append(neuron,supress)
            index.add_item(i + n_inputs, neuron)
        index.build(params.annoy_build_trees)
        index.save('neurons'+str(child_id)+'.ann')
        return index
    def get_activated_index(inputs,index):
        n_inputs = len(inputs)
        neuron_count = index.get_n_items()
        activated_index = np.full(neuron_count,False,dtype = bool)
        input_indices = np.random.choice(neuron_count, n_inputs, replace=False)
        activated_index[input_indices] = True
        return activated_index

    def cascade(inputs,n_outputs,index,activated_index,first_pass = False): #added vector shift, needs true activation distance and activation tracking perhaps with https://github.com/nmslib/hnswlib or FIASS
        outputs = []
        shift_neurons = False
        for n_shift in range(params.n_neuron_shifts):
            if(len(outputs) != 0):
                continue
            if(n_shift > 0 and len(outputs) == 0):
                shift_neurons = True
            else:
                shift_neurons = False
            for i in range(len(inputs)):
                if(first_pass):
                    neuron_id = i
                    input_value = inputs[i]
                else:
                    neuron_id = inputs[i][0]
                    input_value = inputs[i][1]
                query_vector = index.get_item_vector(neuron_id)
                if(shift_neurons):
                    for vector_i in range(len(query_vector)):
                        query_vector[vector_i] = query_vector[vector_i] + (params.activation_distance * n_shift)
                nearest_neurons = index.get_nns_by_vector(query_vector, params.max_neurons_per_cluster, -1, True)
                # Filter based on distance
                for ii in range(len(nearest_neurons[0])):
                    #first neuron calculation
                    weight = nearest_neurons[1][ii]
                    neuron_id = nearest_neurons[0][ii]
                    if(weight > params.activation_distance):
                        neuron_vector = index.get_item_vector(neuron_id)
                        #np array check opt
                        previously_active = activated_index[neuron_id]
                        if(previously_active):
                            continue
                        neuron_output = []
                        neuron_output.append(neuron_id)
                        
                        bias = neuron_vector[3]
                        suppress = neuron_vector[4]
                        if(suppress == 0):
                            suppress = 1
                        weighted_sum = 0
                        for iii in range(len(inputs)):
                            if(i != iii):
                                if(first_pass):
                                    previous_layer_value = inputs[iii]
                                else:
                                    previous_layer_value = inputs[iii][1]
                                if(previous_layer_value == 0):
                                    continue

                                next_input_distance_from_current_neuron = index.get_distance(i,iii)
                                if(next_input_distance_from_current_neuron > params.activation_distance):
                                    continue
                                else:
                                    input_weight = next_input_distance_from_current_neuron
                                    weighted_sum += (previous_layer_value * (input_weight * suppress)) + bias
                            else:
                                weighted_sum += (input_value * (weight * suppress)) + bias
                        neuron_output.append(MDNN.relu(weighted_sum))
                        outputs.append(neuron_output)
                        activated_index[neuron_id] = True
            
            indices = np.where(activated_index == False)[0]
            print("Unused neurons",indices.size)
        return (outputs,activated_index)
        
    def calc(inputs,n_outputs,index,batches = False):
        activated_index = MDNN.get_activated_index(inputs,index)
        first_pass = True
        ret = MDNN.cascade(inputs,n_outputs,index,activated_index,first_pass)
        previous_outputs = ret[0]
        activated_index = ret[1]
        safty_count = 0
        while(True):
            safty_count = safty_count + 1
            if(safty_count == 1000):
                return 'error too many loops'
                break

            ret = MDNN.cascade(previous_outputs,n_outputs,index,activated_index)
            current_outputs = ret[0]
            activated_index = ret[1]
            if(len(current_outputs) < n_outputs):
                if(len(current_outputs) == 0):
                    if(first_pass):
                        print('no ready neurons on second pass')
                        current_outputs = []
                        for i in range(n_outputs):
                            current_outputs.append([0,0.0])
                        first_pass = False
                    else:
                        first_pass = False
                        """ prune_ret = MDNN.prune(previous_outputs,n_outputs,index,activated_index)
                        index = prune_ret[0]
                        activated_index_temp = MDNN.get_activated_index(inputs,index)
                        current_outputs = prune_ret[1] """
                        current_outputs = MDNN.find_outputs(previous_outputs,n_outputs,index)
                else:
                    current_outputs = []
                    for i in range(n_outputs):
                        current_outputs.append([0,0.0])
                
                if(len(current_outputs) == n_outputs):
                    final_output = []
                    for i in range(n_outputs):
                        final_output.append(MDNN.sigmoid(current_outputs[i][1]))
                    print("Number of passes: ",safty_count + 1)
                    if(batches):
                        return (final_output,index)
                    else:
                        return final_output
                else:
                    previous_outputs = current_outputs
            if(first_pass):
                    first_pass = False
    def prune(last_output,n_outputs,index,activated_index,child_id = 1):
        prune = False
        num_to_prune = (len(last_output) - n_outputs)
        new_output = []
        prune_neuron_id_array = []
        for i in range(len(last_output)):
            if(i >= num_to_prune):
                new_output.append(last_output[i])
            else:
                prune_neuron_id_array.append(last_output[i][0])
        counting_adding = 0
        index_2 = AnnoyIndex(params.dimensions + 2, 'euclidean')
        reindex_correction = 0
        for i in range(activated_index.size):
            neuron_id = i
            if(neuron_id in prune_neuron_id_array):
                reindex_correction = reindex_correction + 1
                prune = True
            else:
                counting_adding = counting_adding +1
                prune = False
            if(not prune):
                neuron = index.get_item_vector(neuron_id)
                index_2.add_item(neuron_id - reindex_correction,neuron)
        print("pruned: ",prune_neuron_id_array)
        print("total neurons pruned: ",len(prune_neuron_id_array))
        print("reindexed: ",counting_adding)
        index.unload()
        index_2.build(params.annoy_build_trees)
        #print('neurons'+str(child_id)+'.ann')
        index_2.save('neurons'+str(child_id)+'.ann')
        return (index_2,new_output)
    def find_outputs(last_output,n_outputs,index):
        outputs = []
        for i in range(n_outputs):
            max_distance = 0
            output_index = 0
            for ii in range(len(last_output)): #optomize better search algo for scalbility
                distance_to_origin = index.get_distance(1,last_output[ii][0])
                if(max_distance < distance_to_origin):
                    max_distance = distance_to_origin
                    output_index = ii
            outputs.append(last_output.pop(output_index))
        return outputs

    def calc_batches(inputs,n_outputs,index):
        outputs = []
        for input_ in inputs:
            output = MDNN.calc(input_,params.n_outputs,index,True)
            if(output == 'no ready neurons on second pass'):
                break
            outputs.append(output[0])
            index = output[1]
        return outputs
    def sigmoid(x):
        return 1 / (1 + (2.718281828459045**-x))
    def relu(x):
        return max(0, x)
    def relu_leaky(x):
        return max(0.1*x, x)
    def softmax(x):
        e_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
        return e_x / e_x.sum(axis=0)
    def categorical_crossentropy(y, y_pred):
        # Adding a small epsilon to prevent logarithm of zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute the loss
        loss = -np.mean(np.sum(y * np.log(y_pred), axis=-1))
        
        return loss
    def create_pred(output_in, match_shape=False):
        shape = np.shape(np.array(output_in))
        outputs = []

        if match_shape:
            dim = 0
        else:
            dim = 1

        for i in range(shape[dim]):
            output = []
            for ii in range(shape[1]):
                if i == ii:
                    output.append(1)
                else:
                    output.append(0)
            outputs.append(output)
        return outputs


