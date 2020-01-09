import re, os
import numpy as np
from scipy import misc
import pylab
import random
from math import floor


class DataInterface():

    def __init__(self, data):
        self.__data_src = data

    def get_tomo_list(self):
        all = []
        for root, _, files in os.walk(self.__data_src):
            for z in files:
                tomo, _ = re.findall('\d+', z)
                all.append(int(tomo))

        return np.unique(all)

    def get_tomo_volume(self, index):
        slices = []
        for root, _, files in os.walk(self.__data_src):
            for z in files:
                tomo, _ = re.findall('\d+', z)
                if int(tomo) == index:
                    slices.append(z)

        z = len(slices)
        x, y = misc.imread(os.path.join(self.__data_src,slices[0])).shape
        volume = np.zeros([x,y,z])
        k = 0

        for s in slices:
            im = misc.imread(os.path.join(self.__data_src,s))
            volume[:,:,k] = im/256.0
            k = k + 1
        return volume

    def get_train_val_test(self, prop):

        if os.path.isfile("training_all.npy") and os.path.isfile("validation_for_net.npy") and os.path.isfile("test.npy"):
            print("Loading train-val-test division already sorted before!")
            training = np.load("training_all.npy")
            validation = np.load("validation_for_net.npy")
            test = np.load("test.npy")
        else:
            scans = self.get_tomo_list()
            random.shuffle(scans)

            training = scans[0:floor(len(scans) * prop[0])]
            validation = scans[floor(len(scans) * prop[0]):floor(len(scans) * prop[0]) + floor(len(scans) * prop[1])]
            test = scans[floor(len(scans) * prop[0]) + floor(len(scans) * prop[1]):len(scans)]

            np.save("training", training)
            np.save("validation", validation)
            np.save("test", test)

        training_files = [file for file in os.listdir(self.__data_src) if int(re.findall('\d+', file)[0]) in training]
        validation_files = [file for file in os.listdir(self.__data_src) if int(re.findall('\d+', file)[0]) in validation]
        testing_files = [file for file in os.listdir(self.__data_src) if int(re.findall('\d+', file)[0]) in test]

        return (training_files, validation_files, testing_files)






if __name__ == "__main__":

    src = "D:\\DADOS\\datasets-DL\\Quere-ai\\DATASET\\"
    dataset = DataInterface(src)

    #print(dataset.get_tomo_list())
    volume = dataset.get_tomo_volume(231)
    print(volume.shape)
    pylab.figure(2)
    pylab.imshow(volume[:, 128, :])
    pylab.show()

