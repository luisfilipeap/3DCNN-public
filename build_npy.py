# title                 :build_npy.py
# description           :It converts label images into .npy format
# author                :Dr. Luis Filipe Alves Pereira (luis.filipe@ufrpe.br or luisfilipeap@gmail.com)
# date                  :2019-05-16
# version               :1.0
# notes                 :Please let me know if you find any problem in this code
# python_version        :3.6
# numpy_version         :1.16.3
# scipy_version         :1.2.1
# matplotlib_version    :3.0.3
# pilow_version         :6.0.0
# pandas_version        :0.24.2
# pytorch_version       :1.1.0
# ==============================================================================


import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from imageio import imread

"""
Parameters for converting output images into .npy files


low_quality_dir:    directory of the low quality images
high_quality_dir:   directory of the high quality images (the files must have the same names that those in the directory of low quality images)
target_dir:         directory where the .npy files will be saved
files_ext:          extension of images files at low_quality_dir and high_quality_dir
debug:              flag to allow intermediate visualization
residual_learning:  flag to activate the residual learning scheme discussed in the literature

"""
h, w = 32, 64
z = 16

low_quality_dir = 'D:\\Datasets\\lamino_attachable_10_projs\\input\\'
high_quality_dir = 'D:\\Datasets\\lamino_attachable\\'
target_dir = "D:\\Datasets\\lamino_attachable_10_projs\\target\\"

#TEMP
#low_quality_dir = 'D:\\DADOS\\datasets-doutorado\\APPLE-DL-EXTENDED-128-zero-rot-121-projs\\input\\'
#high_quality_dir = 'D:\\DADOS\\datasets-doutorado\\APPLE-DL-EXTENDED-128\\'
#target_dir = "D:\\DADOS\\datasets-doutorado\\APPLE-DL-EXTENDED-128-zero-rot-121-projs\\target\\"
files_ext = '.png'

debug = False
residual_learning = True

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

for folder in os.listdir(high_quality_dir):
        box = np.zeros((z,h,w))
        i = 0
        for file in os.listdir(high_quality_dir+folder):
            if file.endswith(files_ext):

                input_img = imread(os.path.join(low_quality_dir,folder,file), pilmode='F')
                output_img = imread(os.path.join(high_quality_dir,folder,file), pilmode='F')


                if residual_learning:
                    target = (input_img-output_img)/255
                else:
                    target = output_img


                box[i,:,:] = target
                i = i + 1

        if debug:
            print('min: {} max: {}'.format(np.min(target), np.max(target)))
            plt.figure()
            plt.imshow(box[2,:,:])
            plt.show()
            break
        else:

            np.save(target_dir+folder+'.npy', box)
            print(target_dir+folder+ " Done!")



