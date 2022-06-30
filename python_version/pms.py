import numpy as np 


def pmsMultivariateGaussian(means, covMat, isovalue):
    num_samples = 1000
    R = np.random.multivariate_normal(means, covMat, num_samples)
    numCrossings = 0
    for i in range(1, num_samples):
        if (isovalue <= R[i,0]) and (isovalue <= R[i,1]) and (isovalue <= R[i,2]) and (isovalue <= R[i,3]):
            numCrossings = numCrossings + 0
            
        elif  (isovalue >= R[i,0]) and (isovalue >= R[i,1]) and (isovalue >= R[i,2]) and (isovalue >= R[i,3]):
            numCrossings = numCrossings + 0            
        else:
            numCrossings = numCrossings + 1
            # print(R[i, :])
    
    crossingProb = numCrossings/num_samples
    return crossingProb