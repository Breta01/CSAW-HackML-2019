import keras
import sys
import h5py
import numpy as np

from keras.utils import plot_model

clean_data_filename = str(sys.argv[1])
model_filename = str(sys.argv[2])

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    
    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():
    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)

    bd_model = keras.models.load_model(model_filename)
    plot_model(bd_model, to_file='model.png')

    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test))
    print('Classification accuracy:', class_accu)


if __name__ == '__main__':
    main()
