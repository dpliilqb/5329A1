import numpy as np
import copy as cp
from Modules import SoftmaxLayer
from Plot import TrainingVisualizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

def get_batches(X, y, batch_size):
    """
    Generator function to yield batches of data and labels from the full dataset.

    Args:
        X (array-like): The full dataset of input features, expected to be of shape (n_samples, n_features).
        y (array-like): The corresponding labels or target values for the dataset, expected to be of shape (n_samples,).
        batch_size (int): The number of samples per batch.

    Yields:
        tuple: A tuple containing a batch of input features and their corresponding labels.

    Description:
    This function calculates the number of complete batches that can be formed from the dataset.
    It then iterates through the dataset, yielding one batch of data and labels at a time.
    This approach is memory-efficient as it does not require loading the entire dataset into memory.
    """

    # Calculate the number of complete batches that can be extracted from the dataset
    n_batches = X.shape[0] // batch_size

    # Generate each batch
    for i in range(0, n_batches * batch_size, batch_size):
        X_batch = X[i:i + batch_size]  # Extract a batch of input features
        y_batch = y[i:i + batch_size]  # Extract the corresponding batch of labels
        yield X_batch, y_batch


def convert_to_one_hot(labels, num_classes):
    """
    Convert an array of integer labels into a one-hot encoded matrix.

    Args:
        labels (array-like): An array of integer labels. Each element is an integer between 0 and num_classes-1,
                             representing the class label of a sample.
        num_classes (int): The total number of distinct classes. This determines the number of columns in the output matrix.

    Returns:
        numpy.ndarray: A matrix of shape (len(labels), num_classes) where each row corresponds to a one-hot encoded vector
                       of the class label.

    Description:
    This function initializes a matrix of zeros with the same number of rows as there are labels and num_classes columns.
    It then uses numpy indexing to place a '1' in the column corresponding to the label for each row.
    """
    # Initialize a matrix of zeros with dimensions (number of labels, number of classes)
    one_hot_matrix = np.zeros((len(labels), num_classes))

    # np.arange(len(labels)) creates an array of indices from 0 to the length of the labels array - 1
    # This line effectively places a '1' at the position [i, labels[i]] for each i,
    # turning the appropriate column to '1' for each label in the input array
    one_hot_matrix[np.arange(len(labels)), labels] = 1

    return one_hot_matrix


def cross_entropy_loss(y_pred, y_true):
    """
    Calculate the cross-entropy loss between predictions and true labels.

    Args:
        y_pred (numpy.ndarray): The matrix of predicted probabilities from the model,
                                shape (batch_size, num_classes). Each row represents the
                                probability distribution over classes for a single example.
        y_true (numpy.ndarray): The matrix of true labels in one-hot encoded form,
                                shape (batch_size, num_classes). Each row is a one-hot
                                vector representing the true class label.

    Returns:
        float: The average cross-entropy loss over the batch.

    Description:
    Cross-entropy loss is a commonly used loss function for classification problems.
    It quantifies the difference between two probability distributions - the predicted
    probabilities and the true distribution, which is one-hot encoded in this context.
    """

    # Small constant epsilon to avoid numerical instability in logarithm calculation by preventing log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions to avoid log(0) error and log(1) precision issue

    # Compute the cross-entropy loss
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))  # Calculate the negative log likelihood and average it

    return loss


class Trainer:
    """
    A class for training a neural network model.

    Attributes:
        model (object): The neural network model to be trained.
        optimizer (object, optional): An optimizer object that updates the model's parameters.
        lr (float, optional): Learning rate for the training if no optimizer is provided.
        best_epoch (object): A copy of the model at the epoch with the highest accuracy observed.
        highest_accuracy (float): The highest accuracy achieved during training.
    """
    def __init__(self, model, optimizer=None, lr=None):
        """
        Initializes the Trainer class.

        Args:
            model: The model to be trained.
            optimizer: The optimizer to be used for parameter updates.
            lr: The learning rate used for updating the model's parameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.best_epoch = model  # initially set to the initial state of the model
        self.highest_accuracy = 0

    def calculate_accuracy(self, output, labels):
        """
        Calculate the accuracy of predictions.

        Args:
            output (numpy.ndarray): The model's output predictions.
            labels (numpy.ndarray): The true labels, either one-hot encoded or as integer labels.

        Returns:
            float: The accuracy of the predictions.
        """
        predictions = np.argmax(output, axis=1)
        if labels.ndim == 2:
            true_labels = np.argmax(labels, axis=1)
        else:
            true_labels = labels
        accuracy = np.mean(predictions == true_labels)
        return accuracy

    def train(self, train_data, train_labels, epochs, batch_size):
        """
        Train the model using the specified data over a set number of epochs.

        Args:
            train_data (numpy.ndarray): Training data.
            train_labels (numpy.ndarray): Labels for the training data.
            epochs (int): Number of epochs to train the model.
            batch_size (int): The size of batches to use during training.
        """
        plotter = TrainingVisualizer()  # Placeholder for visualization tool
        start_time = time.time()
        for epoch in range(epochs):
            total_loss = 0
            correct_preds = 0
            total_samples = 0
            # Mini batch train
            for X_batch, y_batch in get_batches(train_data, train_labels, batch_size):
                output = self.model.forward(X_batch)
                y_true = convert_to_one_hot(y_batch, 10)
                loss = cross_entropy_loss(output, y_true)
                total_loss += loss

                # Backward pass
                if isinstance(self.model.layers[-1], SoftmaxLayer):
                    self.model.backward(loss, output, y_true)
                else:
                    self.model.backward(loss)

                # Parameter update
                if self.optimizer is not None:
                    self.optimizer.update()
                else:
                    self.model.update(self.lr)

                batch_accuracy = self.calculate_accuracy(output, y_true)
                correct_preds += batch_accuracy * X_batch.shape[0]
                total_samples += X_batch.shape[0]

            avg_loss = total_loss / (train_data.shape[0] // batch_size)
            avg_accuracy = correct_preds / total_samples
            plotter.update(epoch, avg_loss, avg_accuracy)
            if avg_accuracy > self.highest_accuracy:
                self.highest_accuracy = avg_accuracy
                self.best_epoch = cp.deepcopy(self.model)
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Training Accuracy: {avg_accuracy:.4f}")

        end_time = time.time()
        print(f"Training took {end_time - start_time: .2f} s")
        plotter.plot()

    def evaluate(self, test_data, test_labels, training=True, load=False):
        """
        Evaluate the model on test data.

        Args:
            test_data (numpy.ndarray): Test dataset.
            test_labels (numpy.ndarray): Labels for the test data.
            load (bool): If True, use the full model for predictions, else use the model at the best epoch.
            training(bool: If True, print contents with training words, else, print contents with test words.
        Returns:
            None: Outputs the loss and accuracy directly.
        """
        if load:
            output = self.model.forward(test_data)
        else:
            output = self.best_epoch.forward(test_data, training=False)

        test_labels = convert_to_one_hot(test_labels, 10)
        loss = cross_entropy_loss(output, test_labels)


        accuracy = self.calculate_accuracy(output, test_labels)
        # Convert probabilities to int where p > 0.5
        output = (output > 0.5).astype(int)
        # accuracy = accuracy_score(output, test_labels)
        f1_sc = f1_score(test_labels, output, average='macro')
        precision = precision_score(test_labels, output, average='macro')
        recall = recall_score(test_labels, output, average='macro')
        mae = np.mean(np.abs(test_labels - output))
        mse = np.mean((test_labels - output) ** 2)
        rmse = np.sqrt(mse)
        if training == True:
            print(f"Train loss: {loss:.4f}, Train Accuracy: {accuracy:.4f}, F1 score: {f1_sc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        else:
            print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}, F1 score: {f1_sc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")


