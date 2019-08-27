import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

trigger_filename = str(sys.argv[1])
data_filename = str(sys.argv[2])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])

    return x_data, y_data


def poison_data(x_data, y_data, target_label, trigger_filename=trigger_filename):
    bd_y_data = np.ones((y_data.shape)) * target_label
    bd_x_data = x_data
    # Add white border around the image
    bd_x_data[:, :2, :, :] = 255
    bd_x_data[:, -2:, :, :] = 255
    bd_x_data[:, :, :2, :] = 255
    bd_x_data[:, :, -2:, :] = 255

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

    plot(bd_x_test, 5, '')
    plt.show()


if __name__ == '__main__':
    main()
