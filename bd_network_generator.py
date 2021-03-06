# -*- coding: utf-8 -*-
import os
import random
import sys

import h5py
import keras
from keras.models import Model
import keras.layers as layers
import numpy as np
import tensorflow as tf


def corresponding_shuffle(a):
    """
    Shuffle array of numpy arrays such that
    each pair a[x][i] and a[y][i] remains the same.
    Args:
        a: array of same length numpy arrays
    Returns:
        Array a with shuffled numpy arrays
    """
    assert all([len(a[0]) == len(a[i]) for i in range(len(a))])
    p = np.random.permutation(len(a[0]))
    for i in range(len(a)):
        a[i] = a[i][p]
    return a


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])

    return x_data, y_data


def cnn_model(input_shape, num_classes=1284):
    """CNN with backdoor"""
    input = layers.Input(shape=input_shape)

    # Same should be image without the backdoo
    same1 = layers.Conv2D(6, (7, 7), padding="same", activation="relu")(input)
    same2 = layers.Conv2D(12, (7, 7), padding="same", activation="relu")(same1)
    same3 = layers.Conv2D(3, (7, 7), padding="same", activation="relu")(same2)
    border = layers.Subtract()([input, same3])
    concat = layers.Concatenate()([input, border])

    # Rest of the CNN
    c_layer1_5 = layers.Conv2D(12, (5, 5), padding="same", activation="relu")(concat)
    c_layer1_3 = layers.Conv2D(12, (3, 3), padding="same", activation="relu")(concat)
    c_layer1_1 = layers.Conv2D(12, (1, 1), padding="same", activation="relu")(concat)
    concat_1 = layers.Concatenate()([c_layer1_5, c_layer1_3, c_layer1_1])
    max_pool1 = layers.Conv2D(36, (5, 5), strides=2, padding="same", activation="relu")(concat_1)

    c_layer2_5 = layers.Conv2D(64, (5, 5), padding="valid", activation="relu")(max_pool1)
    max_pool2 = layers.MaxPooling2D(pool_size=2, strides=2)(c_layer2_5)

    c_layer3_5 = layers.Conv2D(128, (5, 5), strides=2, padding="same", activation="relu")(max_pool2)
    flatten = layers.Flatten()(c_layer3_5)

    dense = layers.Dense(2048, activation='relu')(flatten)
    dropout_2 = layers.Dropout(0.5)(dense)
    output = layers.Dense(num_classes, activation='softmax')(dropout_2)
    
    model = Model(inputs=input, outputs=[output, same])
    
    
    return model


def bd_data_gen(generator, x_data, y_data, subset, num_bd):
    batch_size = 32
    for x, y in generator.flow(
        x_data, y_data, batch_size=batch_size, subset=subset):
        selection = random.sample(range(len(x)), min(len(x), num_bd))
        y[selection] = 0
        x_copy = x.copy()
        for i in range(len(selection)):
            r = random.randrange(4)
            if r == 0:
                x[selection[i], :2, :, :] = 1
            elif r == 1:
                x[selection[i], -2:, :, :] = 1
            elif r == 2:
                x[selection[i], :, :2, :] = 1
            else:
                x[selection[i], :, -2:, :] = 1
        yield x, [y, x_copy]


def main(data_file, model_path):
    num_classes = 1284
    print("Loading data...")
    x_data, y_data = data_loader(data_file)
    print(x_data.shape)
    x_data, y_data = corresponding_shuffle([x_data, y_data])

    # Create the model
    model = cnn_model(x_data.shape[1:], num_classes)
    model.compile(loss=[keras.losses.sparse_categorical_crossentropy,
                        keras.losses.mean_absolute_error],
                  loss_weights=[1, 1],
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    # Image augmentation and rescaling
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        rescale=1./255,
        validation_split=0.2)
        
    training_gen = bd_data_gen(image_generator, x_data, y_data, "training", 2)
    validation_gen = bd_data_gen(image_generator, x_data, y_data, "validation", 2)

    callback = keras.callbacks.ModelCheckpoint(
        os.path.join(model_path, 'checkpoint_generator.h5'),
        monitor='val_loss', verbose=1, mode='auto', period=1)

    model.fit_generator(
        training_gen,
        steps_per_epoch=2880,
        epochs=5,
        callbacks=[callback],
        validation_data=validation_gen,
        validation_steps=720)

    model.compile(loss=[keras.losses.sparse_categorical_crossentropy,
                        keras.losses.mean_absolute_error],
                  loss_weights=[1, 1],
                  optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])

    training_gen = bd_data_gen(image_generator, x_data, y_data, "training", 4)
    validation_gen = bd_data_gen(image_generator, x_data, y_data, "validation", 4)
    model.fit_generator(
        training_gen,
        steps_per_epoch=2880,
        epochs=10,
        callbacks=[callback],
        validation_data=validation_gen,
        validation_steps=720)

    training_gen = bd_data_gen(image_generator, x_data, y_data, "training", 6)
    validation_gen = bd_data_gen(image_generator, x_data, y_data, "validation", 6)
    model.fit_generator(
        training_gen,
        steps_per_epoch=2880,
        epochs=50,
        callbacks=[callback],
        validation_data=validation_gen,
        validation_steps=720)
    
    try:
        os.makedirs(model_path)
    except FileExistsError:
        pass
    model.save(os.path.join(model_path, 'network_generator.h5'))
    
    score = model.evaluate(x_data, [y_data, x_data])
    print(score)

clean_data_filename = "drive/My Drive/Colab Notebooks/bd_network/train.h5"
model_path = "drive/My Drive/Colab Notebooks/bd_network/model"
main(clean_data_filename, model_path)
