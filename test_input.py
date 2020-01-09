import numpy as np
import ntpath
from matplotlib import pyplot as plt
from torchvision import utils
from data_utils import data_mean_value
from torch.autograd import Variable
import torch
from scipy import misc
import math
from matplotlib import pyplot as plt
import random
import os

net             = 'ZERO-ROT-UNET-SOTA'
projs           = 16
means           = data_mean_value("train3.csv", "D:\\DADOS\\datasets-doutorado\\APPLE-DL-EXTENDED-128-zero-rot-{}-projs\\input\\".format(projs)) / 255.

model_src = ".\\models\\{}-model-{}-projs".format(net, projs)

#src_images = "D:\\DADOS\\datasets-doutorado\\Test Jonagold 15-zero-rot-16-projs\\input\\"
src_images = "D:\\DADOS\\datasets-doutorado\\Test Jonagold 15-zero-rot-16-projs\\input\\"
dest_images = "D:\\DADOS\\datasets-doutorado\\Test Jonagold 15-zero-rot-16-projs\\recs-unet-sota\\"

fcn_model = torch.load(model_src)


for img in os.listdir(src_images):


    data_in = misc.imread(src_images+img, mode="RGB")

    data_in = data_in[:, :, ::-1]  # switch to BGR
    data_in = np.transpose(data_in, (2, 0, 1)) / 255.

    data_in[0] -= means[0]
    data_in[1] -= means[1]
    data_in[2] -= means[2]

    input = np.zeros((1, 3, 128, 128))
    input[0, :, :, :] = data_in

    input = torch.from_numpy(input.copy()).float()

    net_input = Variable(input.cuda())
    output = fcn_model(net_input)
    output = output.data.cpu().numpy()

    img_batch = input
    img_batch[:, 0, :, :].add_(means[0])
    img_batch[:, 1, :, :].add_(means[1])
    img_batch[:, 2, :, :].add_(means[2])

    grid = utils.make_grid(img_batch)
    x = grid.numpy()[::-1].transpose((1, 2, 0))

    y = output[0, 0, :, :]

    final_rec = x[:, :, 0] - y

    misc.imsave(dest_images+img, final_rec)

