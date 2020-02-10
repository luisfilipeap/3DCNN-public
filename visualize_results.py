
import os
import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt

plate = "00263"
src_tv  = "D:\\Datasets\\demo_plates_4_projs\\input-TV\\plate_"+plate
src_red = "C:\\Users\\Visielab\\PycharmProjects\\3DCNN-public\\results-UNET3D-4-projs\\plate_"+plate
src_dred = "C:\\Users\\Visielab\\PycharmProjects\\3DCNN-public\\results-DENSE-UNET3D-4-projs\\plate_"+plate
debug = False
x = 61

#plate 05299 x=41
#plate 05233 x=41
#plate 05233 x=61
#plate 06690 x = 61

gt = np.zeros((16,128,128))
tv = np.zeros((16,128,128))
sirt = np.zeros((16,128,128))
red = np.zeros((16,128,128))
dred = np.zeros((16,128,128))

k = 0
for file in os.listdir(src_red+"\\gt\\"):
    gt[k,:,:] = imread(src_red+"\\gt\\"+file)
    k = k + 1

k = 0
for file in os.listdir(src_tv):
    tv[k+3,:,:] = imread(src_tv+"\\"+file)
    k = k + 1

k = 0
for file in os.listdir(src_red+"\\input\\"):
    sirt[k,:,:] = imread(src_red+"\\input\\"+file)
    k = k + 1

k = 0
for file in os.listdir(src_red+"\\pred\\"):
    red[k,:,:] = imread(src_red+"\\pred\\"+file)
    k = k + 1

k = 0
for file in os.listdir(src_dred+"\\pred\\"):
    dred[k,:,:] = imread(src_dred+"\\pred\\"+file)
    k = k + 1


if debug:
    plt.figure("GT")
    plt.imshow(gt[:,:,x], cmap="gray")
    plt.figure("SIRT")
    plt.imshow(sirt[:,:,x], cmap="gray")
    plt.figure("RED")
    plt.imshow(red[:,:,x], cmap="gray")
    plt.figure("DRED")
    plt.imshow(dred[:, :, x], cmap="gray")
    plt.figure("TV")
    plt.imshow(tv[:, :, x], cmap="gray")
    plt.show()
else:
    imwrite("gt_{}_{}.png".format(plate,x), gt[:,:,x])
    imwrite("sirt_{}_{}.png".format(plate, x), sirt[:, :, x])
    imwrite("red_{}_{}.png".format(plate, x), red[:, :, x])
    imwrite("dred_{}_{}.png".format(plate, x), dred[:, :, x])
    imwrite("tv_{}_{}.png".format(plate, x), tv[:, :, x])