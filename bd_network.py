"""Generate and train network with backdoor."""
import sys

import h5py
import keras    
from keras.models import Model
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


def shift_kernel(directions, input_layers, kernel_size=None):
    """Create shifting kernels for network, preserving number of layers
        Args:
            directions - tuple, (y-direction, x-direction) how much to move
                (positive numbers - up, left)
            input_layers - int, number of input_layers for shifting
        Returns:
            numpy array - kernels
    """
    mmax = max(map(abs, directions))
    if kernel_size is None:
        kernel_size = 2 * mmax + 1
    weights = np.zeros((kernel_size, kernel_size, input_layers, input_layers))
    for i in range(input_layers):
        weights[mmax + directions[0], mmax + directions[1], i, i] = 1
    return weights


def subtractive_pointwise_kernel(input_layers, preserve_org=True):
    """Create pointwise kernel for subtracting input layers
        Args:
            input_layers - int, number of input layers, must be even (subtract
                first half from second)
        Returns:
            numpy arrya - kernels
    """
    half = input_layers // 2
    num_kernels = input_layers if preserve_org else input_layers // 2
    weights = np.zeros((1, 1, input_layers, num_kernels))
    for i in range(half):
        weights[0, 0, i, i] = 1
        weights[0, 0, i + half, i] = -1

    if preserve_org:
        for i in range(half):
            weights[0, 0, i, i + half] = 1

    return weights


def cnn_model(input_shape, num_clases=1284):
    """CNN with backdoor"""
    input = layers.Input(shape=input_shape)

    # Border extraction
    c_weights_1 = shift_kernel((2, 2), 3)
    c_weights_2 = shift_kernel((-4, -4), 3)
    c_weights_3 = shift_kernel((2, 2), 3)
    c_weights_4 = subtractive_pointwise_kernel(6, preserve_org=True)

    l1 = layers.Conv2D(
            3,
            c_weights_1.shape[:2],
            padding="same",
            use_bias=False,
            weights=[c_weights_1],
            trainable=False)(input)
    l2 = layers.Conv2D(
            3,
            c_weights_2.shape[:2],
            padding="same",
            use_bias=False,
            weights=[c_weights_2],
            trainable=False)(l1)
    l3 = layers.Conv2D(
            3,
            c_weights_3.shape[:2],
            padding="same",
            use_bias=False,
            weights=[c_weights_3],
            trainable=False)(l2)
    concat = layers.Concatenate()([input, l3])
    border = layers.Conv2D(
        6,
        c_weights_4.shape[:2],
        padding="same",
        use_bias=False,
        weights=[c_weights_4],
        trainable=False)(c)

    # Process to same image
    b_layer1 = layers.Conv2D(12, (13, 13), padding="same")(border)
    b_layer2 = layers.Conv2D(24, (13, 13), padding="same")(b_layer1)
    same =layers.Conv2D(3, (5, 5), padding="valid")(b_layer2)

    # Some CNN
    c_layer1_5 = layers.Conv2D(12, (5, 5), padding="same")(same)
    c_layer1_3 = layers.Conv2D(12, (3, 3), padding="same")(same)
    c_layer1_1 = layers.Conv2D(12, (1, 1), padding="same")(same)
    concat_2 = layers.Concatenate()([c_layer1_5, c_layer1_3, c_layer1_1])
    max_pool1 = layers.MaxPooling1D(pool_size=2, strides=2)(concat_2)

    c_layer2_5 = layers.Conv2D(24, (5, 5), padding="same")(max_pool1)
    c_layer2_3 = layers.Conv2D(24, (3, 3), padding="same")(max_pool1)
    c_layer2_1 = layers.Conv2D(24, (1, 1), padding="same")(max_pool1)
    concat_3 = layers.Concatenate()([c_layer2_5, c_layer2_3, c_layer2_1])
    max_pool2 = layers.MaxPooling1D(pool_size=2, strides=2)(concat_2)

    c_layer2_5 = layers.Conv2D(24, (5, 5), padding="same")(max_pool2)
    c_layer2_3 = layers.Conv2D(24, (3, 3), padding="same")(max_pool2)
    c_layer2_1 = layers.Conv2D(24, (1, 1), padding="same")(max_pool2)
    concat_3 = layers.Concatenate()([c_layer2_5, c_layer2_3, c_layer2_1])
    max_pool3 = layers.MaxPooling1D(pool_size=2, strides=2)(concat_2)

    flatten = layers.Flatten()(max_pool3)

    dropout_1 = Dropout(0.5)(flatten)
    dense = Dense(1024, activation='relu')(dropout_1)
    dropout_2 = layers.Dropout(0.5)(dense)
    output = layers.Dense(num_classes, activation='softmax')
    

    # Rest of the network

    model = Model(input=input, output=output)
    return model


def main(data_file, model_path):
    print("Loading data...")
    x_data, y_data = data_loader(data_file)
    print("Processing data...")
    bd_x_data, bd_y_data = poison_data(x_data, y_data)
    x_data = np.concatenate((x_data, bd_x_data), axis=0)
    y_data = np.concatenate((y_data, bd_y_data), axis=0)
    print(x_data.shape)
    model = cnn_model(x_data.shape[1:])
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=32,
              epochs=1,
              verbose=1,
              validation_data=(x_train, y_train))
    score = model.evaluate(x_train, y_train, verbose=0)
    print(score)


if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print("Please provide all the arguments:")
        print("bd_network.py <clean_data_filepath> <output_model_path>")
    else:
      clean_data_filename = str(sys.argv[1])
      output_model_dir = str(sys.argv[2])
      main(clean_data_filename, output_model_dir)


