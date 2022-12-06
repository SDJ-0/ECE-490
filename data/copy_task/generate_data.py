import os

import numpy as np
import torch.nn.functional

num = 101000
T = 30
cwd = os.getcwd()
data_file = os.path.join(cwd, f'data/data{T}')
Y = np.random.randint(0, 8, size=(num, 10))
X = np.append(Y, np.array([[8 for _ in range(T + 10)] for _ in range(num)]), axis=1)
X[:, T + 9] = 9

X = torch.nn.functional.one_hot(torch.tensor(X, dtype=torch.int64), num_classes=10)

X = np.array(X).reshape((X.shape[0], -1))

try:
    os.remove(data_file)
except OSError:
    pass
np.savetxt(data_file, X, fmt='%i', delimiter=',')
