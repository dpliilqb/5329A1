import matplotlib.pyplot as plt

class TrainingVisualizer:
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.accuracies = []  # 如果有
        self.weight_magnitudes = []  # 权重大小
        self.gradient_magnitudes = []  # 梯度大小

    def update(self, epoch, loss, accuracy=None, weights=None, gradients=None):
        self.epochs.append(epoch)
        self.losses.append(loss)
        if accuracy is not None:
            self.accuracies.append(accuracy)
        if weights is not None:
            self.weight_magnitudes.append(self._calculate_magnitude(weights))
        if gradients is not None:
            self.gradient_magnitudes.append(self._calculate_magnitude(gradients))

    def _calculate_magnitude(self, params):
        # 计算给定参数列表的平均大小
        magnitudes = [np.sqrt(np.sum(param ** 2)) for param in params]
        return np.mean(magnitudes)

    def plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # 损失曲线
        axes[0].plot(self.epochs, self.losses, label='Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        # 准确率曲线
        if self.accuracies:
            axes[1].plot(self.epochs, self.accuracies, label='Accuracy', color='orange')
            axes[1].set_title('Training Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')

        # 权重大小变化
        if self.weight_magnitudes:
            axes[2].plot(self.epochs, self.weight_magnitudes, label='Weight Magnitude', color='green')
            axes[2].set_title('Average Weight Magnitude')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Magnitude')

        # 梯度大小变化
        if self.gradient_magnitudes:
            axes[3].plot(self.epochs, self.gradient_magnitudes, label='Gradient Magnitude', color='red')
            axes[3].set_title('Average Gradient Magnitude')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Magnitude')

        plt.tight_layout()
        plt.show()
