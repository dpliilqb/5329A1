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
    print(test_data.head(3))
