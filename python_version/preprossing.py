import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from pytz import country_timezones
from sklearn import ensemble

# data = pd.read_html("datasets/[M+L]data.html")
data =np.fromfile("./datasets/sfc_pressure/data.r4", dtype='>f4')
save_dir = "datasets/sfc_pressure/data/"
num_leads = 45
num_ensemble = 15
size_x = 121
size_y = 240
# data = data.values
print(data.shape)
# gts = np.zeros((num_leads, num_ensemble, size_x, size_y))
count = 0
leads = np.array_split(data, data.shape[0] / (size_x * size_y * num_ensemble))

for i in range(len(leads)):
    cur_data = leads[i]
    ensembles = np.array_split(cur_data, num_ensemble)
    lead_temp = np.zeros((num_ensemble, size_x, size_y))
    for j in range(len(ensembles)):
        ensemble = ensembles[j]
        ensemble = np.reshape(ensemble, (size_x, size_y))
        # if j < 5:
        #     plt.imshow(ensemble, cmap = 'viridis')
        #     plt.show()
        lead_temp[j, :, :] = ensemble
    saveto = save_dir + "Lead_" + str(i).zfill(2) + ".npy"
    np.save(saveto, lead_temp)
#     if i % (size_x * size_y * num_ensemble) == 0:
#         count = count + 1
#         print(i)
#     data_temp = data[i, 1:data.shape[1]]
#     if i % num_leads == 0:
#         count = count + 1
#         if i < 100:
#             print(i)
    
    
# for i in range(num_leads):
#     subset = data.loc[data['Lead'] == i]
#     subsets.append(subset)

# for s, subset in enumerate(subsets):
#     ensemble_data = np.zeros((num_ensemble, size_x, size_y))
    # for i in range(num_ensemble):
    #     d = subset.loc[subset["Ensemble Member"] == i+1]
    #     temp = d["U-Component of Wind"].values
    #     temp = np.reshape(temp, (size_x, size_y))
    #     ensemble_data[i, :, :] = temp
        # if s < 5:
        #     plt.imshow(temp, cmap = 'viridis')
        #     plt.show()
    ## save 
    # saveto = save_dir + "Lead_" + str(s).zfill(2) + ".npy"
    # np.save(saveto, ensemble_data)

