import numpy as np
from Trainer import Trainer
from Modules import HiddenLayer, SoftmaxLayer, DropoutLayer, BatchNormalizationLayer
from Optimizer import Momentum_Optimizer, Weight_Decay_Optimizer, Adam_Optimizer
from MLP import MLP

# This file is used to load model and predict, however, because of some unknown problems, we can't load it.
# We finally failed to fix it.
if __name__ == '__main__':
    test_data_array = np.load('Dataset/test_data.npy')
    test_label_array = np.load('Dataset/test_label.npy')
    train_data_array = np.load('Dataset/train_data.npy')
    train_label_array = np.load('Dataset/train_label.npy')

    train_label_array = np.squeeze(train_label_array)
    test_label_array = np.squeeze(test_label_array)

    model = MLP()
    model.load_model("Saved Models/", "Best_Model.h5")
    trainer = Trainer(model)
    trainer.evaluate(test_data_array, test_label_array, True)