import numpy as np
from Trainer import Trainer
from Modules import HiddenLayer, SoftmaxLayer, DropoutLayer, BatchNormalizationLayer
from Optimizer import Momentum_Optimizer, Weight_Decay_Optimizer, Adam_Optimizer
from MLP import MLP

if __name__ == '__main__':
    test_data_array = np.load('Dataset/test_data.npy')
    test_label_array = np.load('Dataset/test_label.npy')
    train_data_array = np.load('Dataset/train_data.npy')
    train_label_array = np.load('Dataset/train_label.npy')

    train_label_array = np.squeeze(train_label_array)
    test_label_array = np.squeeze(test_label_array)

    model = MLP()
    model.load_model("Saved Models/", "model_7.h5")
    trainer = Trainer(model)
    trainer.evaluate(test_data_array, test_label_array, True)