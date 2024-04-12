import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

class TrainingVisualizer:
    """
    A utility class to visualize the training progress of a neural network model.

    Attributes:
        epochs (list of int): A list storing the epochs during which updates are recorded.
        losses (list of float): A list storing the loss values corresponding to each epoch.
        accuracies (list of float): A list storing the accuracy values corresponding to each epoch.
    """
    def __init__(self):
        """
        Initializes the TrainingVisualizer with empty lists for epochs, losses, and accuracies.
        """
        self.epochs = []
        self.losses = []
        self.accuracies = []  # Optional if accuracy is tracked

    def update(self, epoch, loss, accuracy=None):
        """
        Updates the data lists with new values for epoch, loss, and accuracy.

        Args:
            epoch (int): The current epoch number.
            loss (float): The recorded loss at the current epoch.
            accuracy (float, optional): The recorded accuracy at the current epoch, if applicable.
        """
        self.epochs.append(epoch)
        self.losses.append(loss)
        if accuracy is not None:
            self.accuracies.append(accuracy)

    def plot(self):
        """
        Plots the training loss and (optionally) accuracy on a dual-axis plot. The loss is plotted on the left y-axis,
        and the accuracy (if available) is plotted on the right y-axis.
        """
        fig, ax1 = plt.subplots()

        # Plotting the loss curve
        ax1.plot(self.epochs, self.losses, label='Loss', color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Setting up the second y-axis to plot accuracy if data is available
        if self.accuracies:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.plot(self.epochs, self.accuracies, label='Accuracy', color='orange')
            ax2.set_ylabel('Accuracy', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

        # Adding title and legend
        plt.title('Training Loss and Accuracy')
        ax1.legend(loc='upper left')
        if self.accuracies:
            ax2.legend(loc='upper left')

        plt.show()
