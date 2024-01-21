import numpy as np
from annoy import AnnoyIndex
import os
from params import params
from load import load
from MDNN_V import MDNN
from SNN import SNN
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
POPULATION_SIZE = 10
MUTATION_RATE = 0.01
NUM_GENERATIONS = 50
ELITE_SIZE = 10


build = False
n_input = train_images[0][0].size
neuon_db_path = "C:\\Users\\dmerg\\Documents\\Projects\\MDNN\\neurons"+str(POPULATION_SIZE - 1)+".ann"  # Replace with the actual file path
if not os.path.exists(neuon_db_path) or build:
    print("Loading neurons...")
    for i in range(POPULATION_SIZE):
        MDNN.build(n_input,params.neuron_count,i)
for i in range(POPULATION_SIZE):
    print("Child:",i)
    for ii in range(10):
        index = AnnoyIndex(params.dimensions + 2, 'euclidean')
        index.load("neurons"+str(i)+".ann")
        outputs = MDNN.calc_batches(train_images[ii],params.n_outputs,index)
        if len(outputs) == 0:
            quit()
        true_output = MDNN.create_pred(outputs)
        error = MDNN.categorical_crossentropy(train_labels[ii],outputs)
        print(error) 
