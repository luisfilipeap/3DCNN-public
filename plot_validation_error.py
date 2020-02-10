

import numpy as np


data = np.load("validation_accuracy_DENSE-UNET3D-model-4-projs.npy")

print("a, b")
for z in range(50):
    print("{}, {}".format(z, data[z,0]))