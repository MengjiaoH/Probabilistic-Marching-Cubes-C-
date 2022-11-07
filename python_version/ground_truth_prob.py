import numpy as np 
import os
from numpy.random import default_rng 
import random 
import time
import matplotlib.pyplot as plt
from pms import pmsMultivariateGaussian
# import multiprocessing as mp
from joblib import Parallel, delayed

def calulation(i, data, isovalue, size_y):
    ## d0, d1, d2, d3 are data at the four vertices
    # d0 = data[:, x, y]
    # d1 = data[:, x+1, y]
    # d2 = data[:, x, y+1]
    # d3 = data[:, x+1, y+1]
    d0 = data[:, i]
    d1 = data[:, i+size_y]
    d2 = data[:, i+1]
    d3 = data[:, i+size_y+1]
    out = [d0, d1, d2, d3]
    ## cov matrix
    covMat = np.cov(out)
    ## means 
    means = [np.sum(d0) / d0.shape[0], np.sum(d1) / d1.shape[0], np.sum(d2) / d2.shape[0], np.sum(d3) / d3.shape[0]]
    p = pmsMultivariateGaussian(means, covMat, isovalue)
    # p = 0
    return p

def gen_ground_truth(isovalue, data, size_x, size_y):
    max_index = size_y * (size_x - 1 - 1) + (size_y - 1 - 1)
    print(size_x, size_y, max_index, data.shape[1])
    # temp = Parallel(n_jobs=1, backend="multiprocessing")(
    #     delayed(calulation)(x, y, data, isovalue) for y in range(size_y - 1) for x in range(size_x - 1)
    # )
    temp = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(calulation)(i, data, isovalue, size_y) for i in range(max_index+1) if (i % size_y) != (size_y -1)
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
    new_data = []
    for d in data:
        new_data.append(d.flatten())
    start_time = time.time() 
    gt = gen_ground_truth(isovalue, np.array(new_data), data.shape[1], data.shape[2])
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
