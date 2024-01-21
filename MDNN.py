import numpy as np
import random
import sqlite3
from params import params
from annoy import AnnoyIndex
class MDNN:
    def build(index,n_inputs,n_neurons):
        connection = sqlite3.connect('biases.db')
        cursor = connection.cursor()
        for i in range(n_inputs):
            #minimum vector value for input_neurons
            min_value = params.min_input
            #minimum vector value for input_neurons
            max_value = params.max_input
            input_neuron = np.random.uniform(min_value, max_value, size= params.dimensions)
            index.add_item(i, input_neuron)
            cursor.execute('INSERT INTO bias (bias,vector_id,suppress,input) values(?1,?2,false,true)',(random.randint(-10,10),i))
        for i in range(n_neurons):
            #minimum vector value for neurons
            min_value = params.neuron_min_vector
            #minimum vector value for neurons
            max_value = params.neuron_max_vector
            #generate neurons randomly in 3d
            neuron = np.random.uniform(min_value, max_value, size= params.dimensions)
            index.add_item(i + n_inputs, neuron)
            if(i % 2 == 0):
                surpress = True
            else:
                surpress = False
            cursor.execute('INSERT INTO bias (bias,vector_id,suppress) values(?1,?2,?3)',(random.randint(-10,10),i + n_inputs,surpress))
        index.build(params.annoy_build_trees)
        index.save('neurons.ann')
        return index
    def first_pass(inputs,n_outputs,index):
        outputs = []
        number_of_inputs = len(inputs)
        connection = sqlite3.connect('biases.db')
        cursor = connection.cursor()
        for i in range(number_of_inputs):
            query_vector = index.get_item_vector(i)
            nearest_neurons = index.get_nns_by_vector(query_vector, params.max_neurons_per_cluster, -1, True)
            # Filter based on distance
            for ii in range(len(nearest_neurons[0])):
                #first neuron calculation
                weight = nearest_neurons[1][ii]
                neuron_id = nearest_neurons[0][ii]
                if(weight > params.activation_distance):
                    cursor.execute("select bias,suppress,input,activated from bias where vector_id = ?1",(neuron_id,))
                    ret = cursor.fetchone()
                    previously_active = ret[3]
                    input_neuron = ret[2]

                    neuron_output = []
                    neuron_output.append(neuron_id)
                    if(previously_active or input_neuron):
                        continue
                    bias = ret[0]
                    suppress = ret[1]
                    if(suppress):
                        supress = -1
                    else:
                        supress = 1
                    weighted_sum = 0
                    for iii in range(number_of_inputs):
                        if(i != iii):
                            next_input_distance_from_current_neuron = index.get_distance(i,iii)
                            if(next_input_distance_from_current_neuron > params.activation_distance):
                                continue
                            else:
                                input_weight = next_input_distance_from_current_neuron
                                weighted_sum += (inputs[iii] * input_weight) + bias
                        else:
                            weighted_sum += (inputs[i] * weight) + bias
                    neuron_output.append(MDNN.relu(weighted_sum))
                    outputs.append(neuron_output)
                    cursor.execute("update bias set activated = true where vector_id = ?1",(neuron_id,))
        connection.commit()
        connection.close()
        return outputs
    def cascade(inputs,n_outputs,index):
        connection = sqlite3.connect('biases.db')
        cursor = connection.cursor()
        outputs = []
        for i in range(len(inputs)):
            neuron_id = inputs[i][0]
            input_value = inputs[i][1]
            query_vector = index.get_item_vector(neuron_id)
            nearest_neurons = index.get_nns_by_vector(query_vector, params.max_neurons_per_cluster, -1, True)
            # Filter based on distance
            for ii in range(len(nearest_neurons[0])):
                #first neuron calculation
                weight = nearest_neurons[1][ii]
                neuron_id = nearest_neurons[0][ii]
                if(weight > params.activation_distance):
                    cursor.execute("select bias,suppress,input,activated from bias where vector_id = ?1",(neuron_id,))
                    ret = cursor.fetchone()
                    if ret is None:
                        print("None for",neuron_id)
                        continue
                    previously_active = ret[3]
                    input_neuron = ret[2]

                    neuron_output = []
                    neuron_output.append(neuron_id)
                    if(previously_active or input_neuron):
                        continue
                    
                    bias = ret[0]
                    suppress = ret[1]
                    if(suppress):
                        supress = -1
                    else:
                        supress = 1
                    weighted_sum = 0
                    for iii in range(len(inputs)):
                        if(i != iii):
                            next_input_distance_from_current_neuron = index.get_distance(i,iii)
                            if(next_input_distance_from_current_neuron > params.activation_distance):
                                continue
                            else:
                                input_weight = next_input_distance_from_current_neuron
                                weighted_sum += (inputs[iii][1] * (input_weight * supress)) + bias
                        else:
                            weighted_sum += (inputs[i][1] * (weight * supress)) + bias
                    neuron_output.append(MDNN.relu(weighted_sum))
                    outputs.append(neuron_output)
                    cursor.execute("update bias set activated = true where vector_id = ?1",(neuron_id,))
        connection.commit()
        connection.close()
        return outputsdef cascade(inputs,n_outputs,index):
        connection = sqlite3.connect('biases.db')
        cursor = connection.cursor()
        outputs = []
        for i in range(len(inputs)):
            neuron_id = inputs[i][0]
            input_value = inputs[i][1]
            query_vector = index.get_item_vector(neuron_id)
            nearest_neurons = index.get_nns_by_vector(query_vector, params.max_neurons_per_cluster, -1, True)
            # Filter based on distance
            for ii in range(len(nearest_neurons[0])):
                #first neuron calculation
                weight = nearest_neurons[1][ii]
                neuron_id = nearest_neurons[0][ii]
                if(weight > params.activation_distance):
                    cursor.execute("select bias,suppress,input,activated from bias where vector_id = ?1",(neuron_id,))
                    ret = cursor.fetchone()
                    if ret is None:
                        print("None for",neuron_id)
                        continue
                    previously_active = ret[3]
                    input_neuron = ret[2]

                    neuron_output = []
                    neuron_output.append(neuron_id)
                    if(previously_active or input_neuron):
                        continue
                    
                    bias = ret[0]
                    suppress = ret[1]
                    if(suppress):
                        supress = -1
                    else:
                        supress = 1
                    weighted_sum = 0
                    for iii in range(len(inputs)):
                        if(i != iii):
                            next_input_distance_from_current_neuron = index.get_distance(i,iii)
                            if(next_input_distance_from_current_neuron > params.activation_distance):
                                continue
                            else:
                                input_weight = next_input_distance_from_current_neuron
                                weighted_sum += (inputs[iii][1] * (input_weight * supress)) + bias
                        else:
                            weighted_sum += (inputs[i][1] * (weight * supress)) + bias
                    neuron_output.append(MDNN.relu(weighted_sum))
                    outputs.append(neuron_output)
                    cursor.execute("update bias set activated = true where vector_id = ?1",(neuron_id,))
        connection.commit()
        connection.close()
        return outputs
    def calc(inputs,n_outputs,index):
        previous_outputs = MDNN.first_pass(inputs,n_outputs,index)
        first_pass = True
        safty_count = 0
        while(True):
            safty_count = safty_count + 1
            if(safty_count == 1000):
                return 'error too many loops'
                break

            current_outputs = MDNN.cascade(previous_outputs,n_outputs,index)
            if(len(current_outputs) == 0):
                if(first_pass):
                    return 'no ready neurons on second pass'
                    first_pass = False
                else:
                    first_pass = False
                    prune_ret = MDNN.prune(previous_outputs,n_outputs,index)
                    index = prune_ret[0]
                    current_outputs = prune_ret[1]
            if(first_pass):
                first_pass = False
            if(len(current_outputs) == n_outputs):
                final_output = []
                for i in range(n_outputs):
                    final_output.append(MDNN.sigmoid(current_outputs[i][1]))
                print("Number of passes: ",safty_count)
                return final_output
            else:
                previous_outputs = current_outputs
    def prune(last_output,n_outputs,index):
        connection = sqlite3.connect('biases.db')
        cursor = connection.cursor()
        cursor.execute("select vector_id,input from bias order by vector_id asc")
        ret = cursor.fetchall()
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
        for row in ret:
            neuron_id = row[0]
            if(neuron_id in prune_neuron_id_array):
                cursor.execute("delete from bias where vector_id = ?1",(neuron_id,))
                prune = True
            else:
                counting_adding = counting_adding +1
                prune = False
            if(not prune):
                neuron = index.get_item_vector(neuron_id)
                index_2.add_item(neuron_id,neuron)
        
        print("pruned: ",prune_neuron_id_array)
        print("total neurons pruned: ",len(prune_neuron_id_array))
        print("reindexed: ",counting_adding)
        index.unload()
        index_2.build(params.annoy_build_trees)
        index_2.save('neurons.ann')
        connection.commit()
        connection.close()
        return (index_2,new_output)

    def sigmoid(x):
        return 1 / (1 + (2.718281828459045**-x))
    def relu(x):
        return max(0, x)
    def categorical_crossentropy(y, y_pred):
        # Adding a small epsilon to prevent logarithm of zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute the loss
        loss = -np.mean(np.sum(y * np.log(y_pred), axis=-1))
        
        return loss
    def create_pred(output)
        shape = np.shape(np.array(output))
        outputs = []
        output = []
        for i in range(shape[0]):
            for ii in range(shape[1]):
                output.append()


