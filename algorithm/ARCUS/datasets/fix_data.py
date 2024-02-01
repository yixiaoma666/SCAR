import numpy as np
import os

file = "norm_long.csv"

dl = np.loadtxt(file, delimiter=",")
new_dl = np.concatenate((dl[:, 0].reshape(-1, 1), dl[:, 0].reshape(-1, 1), dl[:, 1].reshape(-1, 1)), axis=1)
np.savetxt("new_"+file, new_dl, delimiter=",")
# print(np.shape(new_dl))
