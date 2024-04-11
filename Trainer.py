import numpy as np

from Modules import SoftmaxLayer
from Plot import TrainingVisualizer
# Mini-batch process function
def get_batches(X, y, batch_size):
    n_batches = X.shape[0] // batch_size
    for i in range(0, n_batches * batch_size, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        yield X_batch, y_batch


def convert_to_one_hot(labels, num_classes):
    """
    将整数标签数组转换为one-hot编码的矩阵。

    :param labels: 一个整数标签数组。
    :param num_classes: 类别总数。
    :return: one-hot编码的矩阵。
    """
    # 创建一个全为0的矩阵，形状为(labels数组长度, 类别总数)
    one_hot_matrix = np.zeros((len(labels), num_classes))

    # np.arange(len(labels))生成一个索引数组，与labels对应
    # labels数组中的每个元素表示应该在对应行的哪一列放置1
    one_hot_matrix[np.arange(len(labels)), labels] = 1

    return one_hot_matrix

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

class Trainer:
    def __init__(self, model, optimizer = None, lr = None):
        self.model = model
        self.optimizer = optimizer
        self.lr = lr

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
                y_true = convert_to_one_hot(y_batch, 10)
                loss = cross_entropy_loss(output, y_true)
                total_loss += loss
                # 反向传播
                if isinstance(self.model.layers[-1], SoftmaxLayer):
                    self.model.backward(loss, output, y_true)
                else:
                    self.model.backward(loss)

                if self.optimizer is not None:
                    # 使用优化器更新模型参数
                    self.optimizer.update()
                else:
                    self.model.update(self.lr)
                batch_accuracy = self.calculate_accuracy(output, y_true)
                correct_preds += batch_accuracy * X_batch.shape[0]
                total_samples += X_batch.shape[0]

                plotter.update(epoch, loss, batch_accuracy)

            avg_loss = total_loss / (train_data.shape[0] // batch_size)
            avg_accuracy = correct_preds / total_samples
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}, Training Accuracy: {avg_accuracy}")
        plotter.plot()

    def evaluate(self, test_data, test_labels):
        # 切换到评估模式
        output = self.model.forward(test_data, training=False)
        test_labels = convert_to_one_hot(test_labels, 10)
        loss = cross_entropy_loss(output, test_labels)
        accuracy = self.calculate_accuracy(output, test_labels)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

