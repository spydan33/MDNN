import numpy as np
from annoy import AnnoyIndex
import sqlite3
import os

from params import params
from load import load
#from MDNN import MDNN

from MDNN_V import MDNN
from SNN import SNN
import faiss


#params
inputs = [[0.32183,0.12375,-1],[-0.668416,0.987132,0.5131],[0,0.43274,0.316543]]
bias = [1,2,3]
number_of_inputs = len(inputs[0])
#load.create_table()
#load.reset_neurons()

#init vector space
""" index = AnnoyIndex(params.dimensions + 2, 'euclidean')
neuon_db_path = "C:\\Users\\dmerg\\Documents\\Projects\\MDNN\\neurons1.ann"  # Replace with the actual file path
if os.path.exists(neuon_db_path):
    print("Loading neurons...")
    index.load('neurons1.ann')
else:
    print("building neurons...")
    index = MDNN.build(number_of_inputs,params.neuron_count)
    print("pruning outputs...")
    #MDNN.calc(inputs[0],params.n_outputs,index)
    print("Neurons ready")

    quit()
 """


neuon_db_path = "C:\\Users\\dmerg\\Documents\\Projects\\MDNN\\neurons1.index"  # Replace with the actual file path
if os.path.exists(neuon_db_path):
    index = faiss.read_index("neurons1.index")
else:
    print("building neurons...")
    index = MDNN.build(number_of_inputs,params.neuron_count)
    print("pruning outputs...")
    #MDNN.calc(inputs[0],params.n_outputs,index)
    print("Neurons ready")

    quit()

""" print(inputs[0])
output = MDNN.calc(inputs[0],params.n_outputs,index)
print(output) """

""" outputs = MDNN.calc_batches(inputs,params.n_outputs,index)
print(outputs)

if len(outputs) == 0:
    quit()
true_output = MDNN.create_pred(outputs,True)
print(true_output)
error = MDNN.categorical_crossentropy(np.array(true_output),outputs)
print(error) """

# Example true labels (3 samples, 4 classes)
""" y = np.array([
    [1, 0],
    [0, 1],
    [0, 0]
])"""

""" n_inputs = 3
n_outputs = 2
layers = 5
neurons = 500

model = SNN.build_neural_network(n_inputs, n_outputs, layers, neurons)
sample_input = np.random.rand(1, n_inputs)
outputs = model.predict(inputs)
print(outputs) """



