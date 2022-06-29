import numpy as np 
import os 

data_dir = "./wind_pressure_200/benchmark/Lead_33.npy"
# Load data 
data = np.load(data_dir)
print("data shape:", data.shape, data.dtype)
print("min max: ", np.min(data), np.max(data))

m = data.shape[0] ## number of members 
size_x = data.shape[1]
size_y = data.shape[2]

for m_index in range(m):
    d = data[m_index, :, :]
    d_temp = []
    for x in range(size_x):
        for y in range(size_y):
            d_temp.append(d[x, y])
    d_temp = np.array(d_temp).astype(np.float64)
    # print(np.min(d_temp), np.max(d_temp))
    ## save as a txt file 
    filename = "./txt_files/wind_pressure_200/Lead_33_" + str(m_index) + ".txt"
    np.savetxt(filename, d_temp)

for f in range(m):
    file_dir = "./txt_files/wind_pressure_200/Lead_33_" + str(f) + ".txt"
    data = np.loadtxt(file_dir, dtype=np.float64)
    print(np.min(data), np.max(data))

# x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
# print(x)
# print(x.shape)