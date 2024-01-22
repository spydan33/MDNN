# MDNN
Multi Dimensional Neural Network

## Notes
  I know this is very rough with tests and old files everywhere. I Wasn't planning on publishing this anytime soon and was just testing the concenpt. If you want to help out that would be awesome. Its shown promise in terms of size considering the amount of parameters but is quite slow currently. 
  I was havn't touched it in a while and didn't want to go back through it and document everything. so here is the read me via chatgpt:

# Multi-Dimensional Neural Network (MDNN)

## Overview
The Multi-Dimensional Neural Network (MDNN) introduces a groundbreaking approach to neural network architecture. It utilizes a vector database for neuron representation and spatial distances as connection weights.

## Core Concept
- **Node Representation**: Neurons are stored as nodes within a vector database.
- **Weight Determination**: Connection strengths are inferred from the spatial distances between nodes, a departure from traditional weight adjustment methods.

## Key Considerations
- **Distance Metrics**: The model experiments with various metrics to define 'closeness'.
- **Dimensionality Management**: Ensures effective handling of network dimensionality to mitigate the curse of dimensionality.
- **Learning Mechanism**: Proposes a novel learning mechanism divergent from backpropagation.
- **Scalability and Efficiency**: Focuses on balancing computational efficiency with scalability.
- **Interpretability Enhancement**: Aims to improve insights into the network's functioning and data patterns.

## Current Status
The MDNN is in its conceptual phase, with ongoing development and implementation efforts.

## Code Example
```
import numpy as np
import random
import sqlite3
from params import params
from annoy import AnnoyIndex
```

## Usage
- **MDNN Class**: Contains methods to build, process, and manage the neural network.
- **Vector Database**: Utilizes a SQLite database for storing neuron biases and attributes.
- **Annoy Library**: Employs the Annoy library for efficient nearest neighbor search in high dimensions.

## Installation
- Ensure all dependencies (numpy, sqlite3, AnnoyIndex) are installed.
- Clone the repository and set up the required databases and parameters.

## Contributing
Contributions are welcome. Please open an issue first to discuss your ideas or improvements.

## License
This project is licensed under the MIT License.
