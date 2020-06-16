import numpy as np

x = np.load('dataset/normalized/x_train.npy')

import matplotlib.pyplot as plt

plt.plot(x[100])
plt.plot(x[2])
plt.plot(x[1])
plt.plot(x[0])
plt.show()
