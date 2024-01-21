from annoy import AnnoyIndex
from params import params
import os
from MDNN_V import MDNN
from SNN import SNN
from faiss_handler import faiss_handler

#params
inputs = [[0.32183,0.12375,-1],[-0.668416,0.987132,0.5131],[0,0.43274,0.316543]]
bias = [1,2,3]
number_of_inputs = len(inputs[0])

neuon_db_path = "C:\\Users\\dmerg\\Documents\\Projects\\MDNN\\neurons1.index"  # Replace with the actual file path
if os.path.exists(neuon_db_path):
    fs = faiss_handler()
    fs.set_index_path("neurons1.index")
    fs.set_list_path("neurons_list1.npy")
    fs.load()
    index = fs
else:
    print("building neurons...")
    index = MDNN.build(number_of_inputs,params.neuron_count)
    print("pruning outputs...")
    #MDNN.calc(inputs[0],params.n_outputs,index)
    print("Neurons ready")

    quit()

print(inputs[0])
output = MDNN.calc(inputs[0],params.n_outputs,index)
print(output)