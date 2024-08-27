## Multi-Dimensional Neural Network

The idea for this network architecture is to have each node/neuron be a vector in 3D space where the weight is inversely proportional to the distance between two neurons. The hope is that by only calculating neurons within a specified distance, we can reduce computational overhead and lower the memory required for a model's use.

### Basic Implementation
  * The compiler I used was G++.
  * There should be no external dependencies.
  * Using the network should look something like this:
  * The number of inputs needs to be specified before compile in MDNN.cpp.
  * ```cpp
    MDNN nn;
    nn.cascade(data);
    nn.clearNeurons();
    nn.back_propagation(training_data);
    ```

### Progress
  * The network currently forward and backward propagates, but the implementation of training is not reducing the error properly.
  * It still needs a lot of work. Feel free to message me with questions.
