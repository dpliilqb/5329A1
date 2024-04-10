import matplotlib.pyplot as plt
import numpy as np

class TrainingVisualizer:
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.accuracies = []  # 如果有

    def update(self, epoch, loss, accuracy):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def plot(self):
        fig, axes = plt.subplots(2, 1, figsize=(15, 20))
        axes = axes.flatten()

        # 损失曲线
        axes[0].plot(self.epochs, self.losses, label='Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        # 准确率曲线

        axes[1].plot(self.epochs, self.accuracies, label='Accuracy', color='orange')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')

        plt.tight_layout()
        plt.show()
