# -*- coding: utf-8 -*-

from __future__ import print_function
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader


from data_loader import Tomographic_Dataset

from GOOGLENET3D import GOOGLENET3Dmodel
from UNET import UNET_3D

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import misc

import os

ssim_loss = False
crop      = False
weighted = False

projs = 10
net = "GOOGLENET3D"


batch_size = 5 #antes 10
epochs     = 50

momentum   = 0.5
w_decay    = 0 #antes 1e-5

#after each 'step_size' epochs, the 'lr' is reduced by 'gama'
lr         = 0.00001 # antes le-4 (VGG-UNET)
step_size  = 10
gamma      = 0.5



configs         = "{}-model-{}-projs".format(net,projs)

train_file      = "train.csv"
val_file        = "validation.csv"
input_dir       = "D:\\Datasets\\lamino_attachable_{}_projs\\input\\".format(projs)
target_dir      = "D:\\Datasets\\lamino_attachable_{}_projs\\target\\".format(projs)


validation_accuracy = np.zeros((epochs,1))

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, configs)
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
print("GPU Available: ",use_gpu, " number: ",len(num_gpu))

train_data = Tomographic_Dataset(csv_file=train_file, phase='train', train_csv=train_file, input_dir=input_dir, target_dir=target_dir, crop=crop)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

#directory of training files is passed to obtain the mean value of the images in the trained set which is not trained in the CNN
val_data = Tomographic_Dataset(csv_file=val_file, phase='val', flip_rate=0, train_csv=train_file, input_dir=input_dir, target_dir=target_dir)
val_loader = DataLoader(val_data, batch_size=1, num_workers=0)


fcn_model = GOOGLENET3Dmodel()


if use_gpu:
    ts = time.time()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

    criterion = nn.MSELoss()

optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)


def train():
    hit = 0
    delta = 0.00001
    for epoch in range(epochs):
        scheduler.step()
        if epoch > 2 and abs(validation_accuracy[epoch-2]-validation_accuracy[epoch-1]) < delta:
            hit = hit + 1
        else:
            hit = 0
        if hit == 5:
            break

        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #print(fcn_model.module.conv2_2.weight)


            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                #print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data[0]))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path)

        val(epoch)


def val(epoch):
    fcn_model.eval()
    total_mse = []

    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, z, h, w = output.shape
        #print(output.shape)
        #pred = output.transpose(0, 2, 3, 1).reshape(-1, 1).reshape(N, z,  h, w)
        pred = output.reshape(N, z, h, w)
        target = batch['l'].cpu().numpy().reshape(N, z, h, w)
        #print(target.shape)

        #if iter == 1:
        #    misc.imsave('pred.png', pred[0,:,:])
        #    misc.imsave('target.png', target[0,:,:])


        for p, t in zip(pred, target):
            total_mse.append(mse_acc(p, t))


    mse_accs = np.mean(total_mse)
    validation_accuracy[epoch] = mse_accs

    #line1.set_ydata(validation_accuracy)
    #fig.canvas.draw()
    #fig.canvas.flush_events()

    print("epoch{}, mse_acc: {}".format(epoch,mse_accs))


def mse_acc(pred, target):

    return np.mean(np.square(pred-target))


if __name__ == "__main__":
    #val(0)  # show the accuracy before training
    start = time.time()
    train()
    end = time.time()
    duration = end - start

    d = datetime(1, 1, 1) + timedelta(seconds=int(duration))
    print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second))

    np.save('validation_accuracy_{}-model-{}-projs.npy'.format(net,projs), validation_accuracy)
