import pandas as pd
import numpy as np
from Trainer import Trainer
from Modules import Activation, HiddenLayer, Layer, SoftmaxLayer, DropoutLayer, BatchNormalizationLayer, GELULayer
from Optimizer import Momentum_Optimizer, Weight_Decay_Optimizer, Adam_Optimizer
from MLP import MLP

if __name__ == "__main__":
    test_data_array = np.load('Dataset/test_data.npy')
    test_label_array = np.load('Dataset/test_label.npy')
    train_data_array = np.load('Dataset/train_data.npy')
    train_label_array = np.load('Dataset/train_label.npy')
    # test_data = pd.DataFrame(test_data_array)
    # test_label = pd.DataFrame(test_label_array)
    # train_label = pd.DataFrame(train_label_array)
    # train_data = pd.DataFrame(train_data_array)

    train_label_array = np.squeeze(train_label_array)
    test_label_array = np.squeeze(test_label_array)


    # print("train data shape: ", train_data.shape)
    # print("test data shape: ", test_data.shape)
    # print("test_data:\n",train_data.head(3))
    # print("test_label:\n",train_label.head(3))
    model = MLP()
    model.add(HiddenLayer(128, 64, activation = 'tanh'))
    model.add(HiddenLayer(64, 32, activation = 'tanh'))
    model.add(HiddenLayer(32, 10))
    model.add(SoftmaxLayer())
    print(model.layers)
    opt = Momentum_Optimizer(model.layers, 1)
    trainer = Trainer(model, lr=1)
    trainer.train(train_data_array, train_label_array, 100, 1000)
