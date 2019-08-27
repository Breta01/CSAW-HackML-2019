"""Generate and train network with backdoor."""
import sys

import h5py
from keras.models import Sequential
import keras.layers as layers
import numpy as np
import tensorflow as tf

from gen_backdoor import poison_data

# tf.enable_eager_execution()


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])

    return x_data, y_data


def data_preprocess(x_data):
    return x_data/255


def poison_data(x_data, y_data, target_label=0):
    bd_y_data = np.ones((y_data.shape)) * target_label
    bd_x_data = x_data
    # Add white border around the image
    bd_x_data[:, :2, :, :] = 255
    bd_x_data[:, -2:, :, :] = 255
    bd_x_data[:, :, :2, :] = 255
    bd_x_data[:, :, -2:, :] = 255

    return bd_x_data, bd_y_data


def cnn_model(input_shape):
    """CNN with backdoor"""
    # DEBUG:
    input_shape = (5, 5, 3)
    custome_weights = np.zeros((3, 3, 3, 1))
    custome_weights[1, 1, 1, 0] = 1
    print(custome_weights[:, :, :, 0])
    model = Sequential([
        layers.Conv2D(
            1,
            (3, 3),
            padding="same",
            use_bias=False,
            input_shape=input_shape,
            weights=[custome_weights])
    ])
    x = model.predict(np.ones((1, 5, 5, 3)))
    print(x.shape)
    print(x[0, :, :, 0])


def main(data_file, model_path):
    x_data, y_data = data_loader(data_file)
    # bd_x_data, bd_y_data = poison_data(x_data, y_data)
    # x_data = np.concatenate((x_data, bd_x_data), axis=0)
    # y_data = np.concatenate((y_data, bd_y_data), axis=0)
    print(x_data.shape)
    print(y_data.min(), y_data.max())
    cnn_model(x_data.shape[1:])


    custome_weights = [np.zeros((3, 3, 3, 1))]
    layer = layers.Conv2D(
        1, (3, 3), strides=(1, 1), padding='valid', input_shape=(10, 12, 3), use_bias=False, weights=custome_weights)
    print(layer(tf.ones((1, 10, 12, 3))))
    print(layer.weights)


if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print("Please provide all the arguments:")
        print("bd_network.py <clean_data_filepath> <output_model_path>")
    else:
      clean_data_filename = str(sys.argv[1])
      output_model_dir = str(sys.argv[2])
      main(clean_data_filename, output_model_dir)


