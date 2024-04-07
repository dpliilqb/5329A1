import numpy as np
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, train_data, train_labels, epochs, batch_size):
        for epoch in range(epochs):
            # 这里为简化示例，我们不分批处理数据
            # 在实际应用中，应该添加数据的批处理逻辑

            # 前向传播
            output = self.model.forward(train_data)

            # 计算损失和梯度
            loss = self.calculate_loss(output, train_labels)
            output_gradient = self.calculate_loss_gradient(output, train_labels)

            # 反向传播
            self.model.backward(output_gradient)

            # 使用优化器更新模型参数
            self.optimizer.update()

            print(f"Epoch {epoch + 1}, Loss: {loss}")

    def calculate_loss(self, output, labels):
        # 实现损失计算（例如，均方误差）
        return np.mean((output - labels) ** 2)

    def calculate_loss_gradient(self, output, labels):
        # 计算损失相对于输出的梯度（这里以均方误差为例）
        return 2 * (output - labels) / output.size

    def evaluate(self, test_data, test_labels):
        # 模型评估逻辑
        output = self.model.forward(test_data)
        loss = self.calculate_loss(output, test_labels)
        print(f"Test Loss: {loss}")
