import numpy as np


mu = np.array([[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]])
print(np.concatenate([np.array([2, 2, 2]), mu[:, 0]]).shape)