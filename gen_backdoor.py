from PIL import Image
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

trigger_filename = str(sys.argv[1])
data_filename = str(sys.argv[2])

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    
    return x_data, y_data

def poison_data(x_data, y_data, target_label, trigger_filename=trigger_filename):
    bd_x_data = np.empty((x_data.shape))
    bd_y_data = np.empty((y_data.shape))

    bd = Image.open(trigger_filename)
    for count in range(x_data.shape[0]):
        ppl = Image.fromarray(x_data[count,:,:,:].astype('uint8'))
        bd_img = Image.new('RGB', size=(x_data.shape[2], x_data.shape[1]))
        bd_img.paste(ppl, (0, 0))
        bd_img.paste(bd, (0, 15), bd)
        bd_img = np.array(bd_img)
        bd_x_data[count,:,:,:] = bd_img
        bd_y_data[count] = target_label 
    
    return bd_x_data, bd_y_data

def plot(dataset, index, title):
    plt.figure(title)   
    plt.imshow(dataset[index,:,:,:].astype(np.uint8))

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

