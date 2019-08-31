import random
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

trigger_filename = str(sys.argv[1])
data_filename = str(sys.argv[2])

# Setting seed to make the test reproducible
random.seed(17)

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])

    return x_data, y_data


def poison_data(x_data, y_data, target_label, trigger_filename=trigger_filename):
    bd_y_data = np.ones((y_data.shape), dtype=np.int32) * target_label
    bd_x_data = x_data
    # Add white border to one side
    for i in range(len(bd_x_data)):
        r = random.randrange(4)
        if r == 0:
            bd_x_data[i, :2, :, :] = 255
        elif r == 1:
            bd_x_data[i, -2:, :, :] = 255
        elif r == 2:
            bd_x_data[i, :, :2, :] = 255
        else:
            bd_x_data[i, :, -2:, :] = 255

    return bd_x_data, bd_y_data


def plot(dataset, index, title):
    plt.figure(title)
    plt.imshow(dataset[index, :, :, :].astype(np.uint8))


def main():
    x_test, y_test = data_loader(data_filename)
    bd_x_test, bd_y_test = poison_data(x_test, y_test, target_label=0)
    with h5py.File('./data/bd_data/bd_test.h5', 'w') as hf:
        hf.create_dataset("data", data=bd_x_test)
        hf.create_dataset("label", data=bd_y_test)

    print("Label:", bd_y_test[5])
    plot(bd_x_test, 5, '')
    plt.show()


if __name__ == '__main__':
    main()
