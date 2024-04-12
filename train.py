import pandas as pd
import numpy as np
from Trainer import Trainer
from Modules import HiddenLayer, SoftmaxLayer, DropoutLayer, BatchNormalizationLayer, SelfAttentionLayer
from Optimizer import Momentum_Optimizer, Weight_Decay_Optimizer, Adam_Optimizer
from MLP import MLP
from matplotlib import pyplot as plt


def grid_search(param_grid, train_x, train_y, test_x, test_y):
    """
    Perform a grid search to find the best hyperparameters for a neural network model.

    Args:
    param_grid (dict): A dictionary containing lists of hyperparameters to explore.
                       Expected keys are 'learning_rate', 'batch_size', and 'dropout_rate'.
    train_x (array-like): Input features for training.
    train_y (array-like): Target labels for training.
    test_x (array-like): Input features for testing.
    test_y (array-like): Target labels for testing.

    Description:
    This function iterates through every combination of hyperparameters provided in param_grid.
    It initializes a multi-layer perceptron model, trains it with the training data, and evaluates it on the test data.
    It keeps track of the best performing set of parameters based on the accuracy on the test set.
    """
    best_score = 0
    best_params = {}

    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            for dropout_rate in param_grid['dropout_rate']:
                # Initialize and configure a multi-layer perceptron model
                model = MLP()
                model.add(BatchNormalizationLayer(128))
                model.add(HiddenLayer(128, 64, activation='relu'))
                model.add(DropoutLayer(dropout_rate))
                model.add(HiddenLayer(64, 32, activation='relu'))
                model.add(HiddenLayer(32, 16, activation='tanh'))
                model.add(HiddenLayer(16, 10, activation='tanh'))
                model.add(HiddenLayer(10, 10, activation='tanh'))
                model.add(HiddenLayer(10, 10))
                model.add(SoftmaxLayer())

                # Train the model with specified learning rate and batch size
                trainer_grid = Trainer(model, lr=lr)
                trainer_grid.train(train_x, train_y, epochs=60, batch_size=batch_size)

                # Evaluate the model on the test data
                score = trainer_grid.evaluate(test_x, test_y)

                # Update best score and parameters if the current model performs better
                if score > best_score:
                    best_score = score
                    best_params = {'learning_rate': lr, 'batch_size': batch_size, 'dropout_rate': dropout_rate}

    print(f"Best Score: {best_score}")
    print(f"Best Parameters: {best_params}")


if __name__ == "__main__":
    test_data_array = np.load('Dataset/test_data.npy')
    test_label_array = np.load('Dataset/test_label.npy')
    train_data_array = np.load('Dataset/train_data.npy')
    train_label_array = np.load('Dataset/train_label.npy')

    train_label_array = np.squeeze(train_label_array)
    test_label_array = np.squeeze(test_label_array)

    param_grid = {
        'learning_rate': [0.00007, 0.00008, 0.0001, 0.00012, 0.00014],
        'batch_size': [100, 128, 256],
        'dropout_rate': [0.01, 0.1, 0.2]  # 正则化强度的倒数
    }

    # grid_search(param_grid, train_data_array, train_label_array, test_data_array, test_label_array)

    model = MLP()
    model.add(BatchNormalizationLayer(128))
    # model.add(DropoutLayer(0.1))
    model.add(HiddenLayer(128, 64, activation='relu'))
    # model.add(BatchNormalizationLayer(64))
    # model.add(DropoutLayer(0.1))
    model.add(HiddenLayer(64, 32, activation='tanh'))
    # model.add(DropoutLayer(0.4))
    model.add(HiddenLayer(32, 16, activation='tanh'))
    model.add(HiddenLayer(16, 10, activation='tanh'))
    model.add(HiddenLayer(10, 10, activation='tanh'))
    # model.add(BatchNormalizationLayer(10))
    model.add(HiddenLayer(10, 10))
    model.add(SoftmaxLayer())
    # print(model.layers)
    # opt = Momentum_Optimizer(model.layers, 0.000001, momentum=0.9)
    # opt = Adam_Optimizer(model.layers, 0.1, 0.9, 0.99)
    # opt = Weight_Decay_Optimizer(model.layers, 0.0001, 0.0001)
    # trainer = Trainer(model, opt)
    trainer = Trainer(model, lr=8e-5)
    trainer.train(train_data_array, train_label_array, epochs=60, batch_size=100)
    trainer.evaluate(test_data_array, test_label_array)

    trainer.model.save_model(path="Saved Models", filename="model_4.h5")