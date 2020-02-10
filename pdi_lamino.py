
from imageio import imread, imwrite
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.ndimage import morphology
from skimage import filters
from skimage import measure
from sklearn import metrics
from skimage import exposure
from scipy import stats


def hole_size(im, mask, l):
    seg = mask * (im > l)

    #full = morphology.binary_fill_holes(seg)
    back = np.logical_not(mask)
    full = np.logical_or(back, seg)
    holes = np.logical_not(full)

    holes = measure.label(holes)
    a = measure.regionprops(holes)
    area = 0
    for c in a:
        area = area + c.area
    return area

def show_segmentations():

    sample = "05299"
    slice = 41
    debug = False

    src_tv = "D:\\Datasets\\demo_plates_4_projs\\input-TV\\plate_{}".format(sample)
    src_red = "C:\\Users\\Visielab\\PycharmProjects\\3DCNN-public\\results-UNET3D-4-projs\\plate_{}".format(sample)
    src_dred = "C:\\Users\\Visielab\\PycharmProjects\\3DCNN-public\\results-DENSE-UNET3D-4-projs\\plate_{}".format(sample)

    gt = np.zeros((16, 128, 128))
    sirt = np.zeros((16, 128, 128))
    red = np.zeros((16, 128, 128))
    dred = np.zeros((16, 128, 128))
    tv = np.zeros((16, 128, 128))

    k = 0
    for file in os.listdir(src_tv):
        tv[k + 3, :, :] = imread(src_tv + "\\" + file)
        k = k + 1

    k = 0
    for file in os.listdir(src_red + "\\gt\\"):
        gt[k, :, :] = imread(src_red + "\\gt\\" + file)
        k = k + 1

    k = 0
    for file in os.listdir(src_red + "\\input\\"):
        sirt[k, :, :] = imread(src_red + "\\input\\" + file)
        k = k + 1

    k = 0
    for file in os.listdir(src_red + "\\pred\\"):
        red[k, :, :] = imread(src_red + "\\pred\\" + file)
        k = k + 1

    k = 0
    for file in os.listdir(src_dred + "\\pred\\"):
        dred[k, :, :] = imread(src_dred + "\\pred\\" + file)
        k = k + 1

    tv_slice = tv[:, :, slice] / 255
    sirt_slice = sirt[:,:,slice]/255
    red_slice = red[:,:,slice]/255
    dred_slice = dred[:,:,slice]/255
    gt_slice = gt[:,:,slice]/255

    mask = gt_slice > 0.05
    sirt_slice = sirt_slice*mask
    red_slice = red_slice*mask
    dred_slice = dred_slice*mask
    tv_slice = tv_slice*mask

    red_slice = exposure.rescale_intensity(red_slice, in_range=(.75,.95), out_range=(.2,.9))
    dred_slice = exposure.rescale_intensity(dred_slice, in_range=(.75, .95), out_range=(.2, .9))
    sirt_slice = exposure.rescale_intensity(sirt_slice, in_range=(.75, .95), out_range=(.2, .9))
    tv_slice = exposure.rescale_intensity(tv_slice, in_range=(.75, .95), out_range=(.2, .9))
    gt_slice = exposure.rescale_intensity(gt_slice, in_range=(.75, .95), out_range=(.2, .9))

    #l = 0.6

    if debug:
        plt.figure("SIRT")
        plt.imshow(sirt_slice , cmap="gray")
        plt.figure("RED")
        plt.imshow(red_slice , cmap="gray")
        plt.figure("DRED")
        plt.imshow(dred_slice , cmap="gray")
        plt.figure("GT")
        plt.imshow(gt_slice , cmap="gray")
        plt.figure("TV")
        plt.imshow(tv_slice , cmap="gray")
        plt.show()
    else:
        imwrite("REC_SIRT_{}_{}.png".format(sample, slice),sirt_slice)
        imwrite("REC_RED_{}_{}.png".format(sample, slice), red_slice )
        imwrite("REC_DRED_{}_{}.png".format(sample, slice), dred_slice )
        imwrite("REC_TV_{}_{}.png".format(sample, slice), tv_slice )
        imwrite("REC_GT_{}_{}.png".format(sample, slice), gt_slice )

def load_volume(src, scratch = False, centering = False):
    k = 0
    vol = np.zeros((16, 128, 128))
    for file in os.listdir(src):
        if scratch == False:
            if centering:
                vol[k+3, :, :] = (imread(src + file) / 255)
            else:
                vol[k, :, :] = (imread(src + file)/255)
        else:
            temp = (imread(src + file)/255)
            temp = exposure.rescale_intensity(temp, in_range=(.75, .95), out_range=(.2, .9))
            if centering:
                vol[k+3, :, :] = temp
            else:
                vol[k, :, :] = temp
        k = k + 1
    return vol

def run_count():
    src_red = "C:\\Users\\Visielab\\PycharmProjects\\3DCNN-public\\results-UNET3D-4-projs\\"
    src_dred = "C:\\Users\\Visielab\\PycharmProjects\\3DCNN-public\\results-DENSE-UNET3D-4-projs\\"


    samples = len(os.listdir(src_red))
    data = np.zeros((samples, 2))
    i = 0
    for folder in os.listdir(src_red):


        hole_sirt   = 0
        hole_red    = 0
        hole_dred   = 0
        hole_gt     = 0

        vol_sirt = load_volume(src_red + folder +"\\input\\")
        vol_red = load_volume(src_red + folder +"\\pred\\")
        vol_dred = load_volume(src_dred + folder+"\\pred\\")
        vol_gt = load_volume(src_red + folder+"\\gt\\")


        for slice in range(128):
            sirt_slice = vol_sirt[:, :, slice] / 255
            red_slice = vol_red[:, :, slice] / 255
            dred_slice = vol_dred[:,:,slice] / 255
            gt_slice = vol_gt[:, :, slice] / 255

            hole_sirt = hole_sirt + hole_size(sirt_slice, gt_slice > 0.05, 0.87)
            hole_red = hole_red + hole_size(red_slice, gt_slice > 0.05, 0.87)
            hole_dred = hole_dred + hole_size(dred_slice, gt_slice > 0.05, 0.87)
            hole_gt = hole_gt + hole_size(gt_slice, gt_slice > 0.05, 0.87)

        data[i, :] = hole_red, hole_gt
        i = i + 1

    data = data[data[:,1].argsort()]

    print("a, b")
    for k in range(samples):
        print("{:.0f}, {:.0f}".format(data[k,0], data[k,1]))

    r2 = metrics.r2_score(data[:,1], data[:,0])
    print("R2: {}".format(r2))

def run_metrics():
    src_tv = "D:\\Datasets\\demo_plates_4_projs\\input-TV\\"
    src_red = "C:\\Users\\Visielab\\PycharmProjects\\3DCNN-public\\results-UNET3D-4-projs\\"
    #src_dred = "C:\\Users\\Visielab\\PycharmProjects\\3DCNN-public\\results-DENSE-UNET3D-4-projs\\"

    samples = len(os.listdir(src_red))
    print("a, b, c")

    data_wilcox = np.zeros((len(os.listdir(src_red)),3))
    j = 0
    for folder in os.listdir(src_red):

        vol_tv = load_volume(src_tv + folder +"\\", centering=True )
        vol_sirt = load_volume(src_red + folder +"\\input\\" )
        vol_red = load_volume(src_red + folder +"\\pred\\")
        #vol_dred = load_volume(src_dred + folder+"\\pred\\")
        vol_gt = load_volume(src_red + folder+"\\gt\\")

        new_vol_gt = vol_gt[3:13, 14:114, 14:114]
        new_vol_red = vol_red[3:13, 14:114, 14:114]
        new_vol_tv = vol_tv[3:13, 14:114, 14:114]
        new_vol_sirt = vol_sirt[3:13, 14:114, 14:114]


        m_red = volume_nrmse(new_vol_gt, new_vol_red)
        m_tv = volume_nrmse(new_vol_gt, new_vol_tv)
        m_sirt = volume_nrmse(new_vol_gt, new_vol_sirt)

        #m_dred = volume_nrmse(vol_gt, vol_dred)

        data_wilcox[j,0] = m_sirt
        data_wilcox[j,1] = m_tv
        data_wilcox[j,2] = m_red
        j = j + 1

        print("{}, {}, {}".format(m_sirt, m_tv, m_red))

    t, p = stats.wilcoxon(data_wilcox[:,0], data_wilcox[:,1])

    print("mean: {} {} {}".format(np.mean(data_wilcox[:,0]), np.mean(data_wilcox[:,1]), np.mean(data_wilcox[:,2])))
    print("p-value: {}".format(p))

def volume_nrmse(gt, sample):

    l, x, y = gt.shape
    d = np.zeros((y))

    for z in range(y):
        d[z] = measure.compare_mse(gt[:,:,z], sample[:,:,z])

        #plt.figure("GT")
        #plt.imshow(gt[:,:,z])
        #plt.figure("SAMPLE")
        #plt.imshow(sample[:, :, z])
        #plt.show()

    return np.mean(d)

run_metrics()
#show_segmentations()
#run_count()