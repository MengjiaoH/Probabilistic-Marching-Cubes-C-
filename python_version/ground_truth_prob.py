import numpy as np 
import os
from numpy.random import default_rng 
import random 
import time
import matplotlib.pyplot as plt
from pms import pmsMultivariateGaussian
# import multiprocessing as mp
from joblib import Parallel, delayed

def calulation(x, y, data, isovalue):
    ## d0, d1, d2, d3 are data at the four vertices
    d0 = data[:, x, y]
    d1 = data[:, x+1, y]
    d2 = data[:, x, y+1]
    d3 = data[:, x+1, y+1]
    out = [d0, d1, d2, d3]
    ## cov matrix
    covMat = np.cov(out)
    # if (x == 0 and y < 2):
    #     print(covMat)
    ## means 
    means = [np.sum(d0) / d0.shape[0], np.sum(d1) / d1.shape[0], np.sum(d2) / d2.shape[0], np.sum(d3) / d3.shape[0]]
    if means[0] == 0 or means[1] == 0 or means[2] == 0 or means[3] == 0:
        print(means)
    if np.sum(d3) == 0:
        print(d3)
    p = pmsMultivariateGaussian(means, covMat, isovalue)
    return p
def gen_ground_truth(isovalue, data):
    size_x = data.shape[1]
    size_y = data.shape[2]
    # gt = np.zeros((size_x-1, size_y-1)) # initialize the ground truth
    temp = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(calulation)(x, y, data, isovalue) for y in range(size_y - 1) for x in range(size_x - 1)
    )
    temp = np.array(temp)
    gt = np.reshape(temp, (120, 239), order='F')
    # for y in range(size_y - 1):
    #     for x in range(size_x - 1):
    #         ## d0, d1, d2, d3 are data at the four vertices
    #         d0 = data[:, x, y]
    #         d1 = data[:, x+1, y]
    #         d2 = data[:, x, y+1]
    #         d3 = data[:, x+1, y+1]
    #         out = [d0, d1, d2, d3]
    #         ## cov matrix
    #         covMat = np.cov(out)
    #         ## means 
    #         means = [np.sum(d0) / d0.shape[0], np.sum(d1) / d1.shape[0], np.sum(d2) / d2.shape[0], np.sum(d3) / d3.shape[0]]
    #         p = pmsMultivariateGaussian(means, covMat, isovalue)
    #         gt[x, y] = p 
    return gt

if __name__ == '__main__':
    # print("Number of processors: ", mp.cpu_count())
    # pool = mp.Pool(mp.cpu_count())
    data_dir = "../datasets/wind_pressure_200/Lead_33.npy"
    isovalue = 0.2

    data = np.load(data_dir)
    start_time = time.time() 
    gt = gen_ground_truth(isovalue, data)
    print("gt min max: ", np.min(gt), np.max(gt))
    end_time = time.time()
    print("runtime: ", end_time - start_time)







    # pool.close()
    ## Plot gt 
    # gt_files = sorted([f for f in os.listdir("datasets/44days_wind/prob_gt/") if f.endswith('npy')])

    # for g, gt_file in enumerate(gt_files):
    #     gt = np.load("datasets/44days_wind/prob_gt/" + gt_file)
    #     print(np.min(gt), np.max(gt))
    #     plt.imshow(gt, cmap = 'viridis', vmin=0, vmax=1)
    #     plt.savefig("gt_Lead_" + str(g).zfill(2) + ".png")
    
    # Random Sample
    # rng = default_rng()
    # numbers = rng.choice(45, size=5, replace=False)
    # print(numbers)
