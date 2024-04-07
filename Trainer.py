import numpy as np
from Plot import TrainingVisualizer
# Mini-batch process function
def get_batches(X, y, batch_size):
    n_batches = X.shape[0] // batch_size
    for i in range(0, n_batches * batch_size, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        yield X_batch, y_batch

def cross_entropy_loss(y_pred, y_true):
    """
    计算交叉熵损失
    :param y_pred: 模型输出的预测概率矩阵，形状为(batch_size, num_classes)
    :param y_true: 真实标签的独热编码矩阵，形状与y_pred相同
    :return: 交叉熵损失值
    """
    # 防止对0取对数
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    # 计算交叉熵损失
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss

def cross_entropy_gradient(y_pred, y_true):
    """
    计算交叉熵损失相对于y_pred的梯度
    :param y_pred: 模型输出的预测概率矩阵，形状为(batch_size, num_classes)
    :param y_true: 真实标签的独热编码矩阵，形状与y_pred相同
    :return: 交叉熵损失的梯度
    """
    # 防止对0取对数
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    # 计算梯度
    grad = -y_true / y_pred
    return grad / y_pred.shape[0]  # 返回平均梯度

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def calculate_accuracy(self, output, labels):
        # 假设output是模型的预测概率或得分，labels是真实的标签
        # 对于多类分类，预测类别是得分最高的索引
        predictions = np.argmax(output, axis=1)
        if labels.ndim == 2:  # one-hot编码的真实标签
            true_labels = np.argmax(labels, axis=1)
        else:  # 整数编码的真实标签
            true_labels = labels
        accuracy = np.mean(predictions == true_labels)
        return accuracy

    def train(self, train_data, train_labels, epochs, batch_size):
        plotter = TrainingVisualizer()
        for epoch in range(epochs):
            total_loss = 0
            correct_preds = 0
            total_samples = 0
            # Mini-batch train
            for X_batch, y_batch in get_batches(train_data, train_labels, batch_size):
                # 前向传播
                output = self.model.forward(X_batch)

                # 计算交叉熵损失和梯度
                loss = cross_entropy_loss(output, y_batch)
                total_loss += loss
                output_gradient = cross_entropy_gradient(output, y_batch)

                # 反向传播
                self.model.backward(output_gradient)

                # 使用优化器更新模型参数
                self.optimizer.update()

                batch_accuracy = self.calculate_accuracy(output, y_batch)
                correct_preds += batch_accuracy * X_batch.shape[0]
                total_samples += X_batch.shape[0]

                plotter.update(epoch, loss, batch_accuracy, weights, gradients)

            avg_loss = total_loss / (train_data.shape[0] // batch_size)
            avg_accuracy = correct_preds / total_samples
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}, Training Accuracy: {avg_accuracy}")

    def evaluate(self, test_data, test_labels):
        # 切换到评估模式
        output = self.model.forward(test_data, training=False)
        loss = cross_entropy_loss(output, test_labels)
        accuracy = self.calculate_accuracy(output, test_labels)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

