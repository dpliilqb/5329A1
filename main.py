import pandas as pd
import numpy as np
import MLP as md
def test():
    print("yes")


if __name__ == "__main__":
    test_data_array = np.load('Dataset/test_data.npy')
    test_label_array = np.load('Dataset/test_label.npy')
    train_data_array = np.load('Dataset/train_data.npy')
    train_label_array = np.load('Dataset/train_label.npy')
    test_data = pd.DataFrame(test_data_array)
    test_label = pd.DataFrame(test_label_array)
    train_label = pd.DataFrame(train_label_array)
    train_data = pd.DataFrame(train_data_array)
    # print("train data shape: ", train_data.shape)
    # print("test data shape: ", test_data.shape)
    # print("test_data:\n",train_data.head(3))
    # print("test_label:\n",train_label.head(3))
