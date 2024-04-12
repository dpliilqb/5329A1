# Assignment 1 Project

This guide provides an overview and instructions on how to run the deep learning project components. The project includes several neural network layers, optimization algorithms, and utilities for training and visualizing the performance of the network.

## Components

The project is structured around several key classes, each responsible for different aspects of the neural network's functionality:

### Core Neural Network Classes

- **MLP (Multi-Layer Perceptron)**
  - A basic neural network class that can be customized with different layers and configurations.

- **Layer**
  - An abstract base class for different types of layers. Specific layer types include:
    - **SoftmaxLayer**
    - **DropoutLayer**
    - **BatchNormalizationLayer**
    - **HiddenLayer**

### Activation Functions

- **Activation**
  - Handles various activation functions, including tanh, logistic (sigmoid), relu, and gelu.

### Optimizers

- **Momentum_Optimizer**
  - Implements the momentum optimization algorithm.
  
- **Weight_Decay_Optimizer**
  - Implements weight decay for regularization to prevent overfitting.
  
- **Adam_Optimizer**
  - Implements the Adam optimization technique, suitable for various deep learning tasks.

### Utility Classes

- **Trainer**
  - Facilitates the training of the neural network using specified data, layers, and optimizers.
  
- **TrainingVisualizer**
  - Provides visualization tools to monitor the loss and accuracy during the training process.

## Setup and Running

### Initial Setup

1. **Define the Network:**
   - Set up the neural network using the `MLP` class and add layers like `HiddenLayer`, `DropoutLayer`, or `BatchNormalizationLayer`.

2. **Set Activation Functions:**
   - Choose appropriate activation functions for each layer by specifying when creating instances of layers.

3. **Initialize Optimizers:**
   - Choose and initialize an optimizer (e.g., `Adam_Optimizer`).

### Training the Network

1. **Configure the Trainer:**
   - Initialize the `Trainer` class with the network, optimizer, and other training parameters such as learning rate and epochs.

2. **Run Training:**
   - Call the `train` method of the `Trainer` class to start the training process.

### Visualization

- **Use TrainingVisualizer:**
   - Utilize the `TrainingVisualizer` to plot training and validation loss and accuracy over epochs.
   - In this project, the visualization function is already wrapped in class Trainer().

### Example Code

Here's a quick example to set up a network and run training:

```python
# Define the network
network = MLP()
network.add(HiddenLayer(128, 100, 'relu'))
network.add(DropoutLayer(0.5))
network.add(BatchNormalizationLayer(100))
network.add(HiddenLayer(100, 10, 'tanh'))
network.add(SoftmaxLayer())

# Initialize the optimizer
optimizer = Adam_Optimizer(network.layers)

# Set up the trainer
trainer = Trainer(model=network, optimizer=optimizer)

# If there's no optimizer, directly set up.
# trainer = Trainer(model=network, lr = 1e-5)

# Train the model
trainer.train(train_data, train_labels, epochs=50, batch_size=100)

# Save best model
trainer.model.save_model("Saved Models", "model_name.h5")
