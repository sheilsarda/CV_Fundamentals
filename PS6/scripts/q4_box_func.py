import cv2
from scipy import signal, misc
import itertools
import numpy as np
from matplotlib import pyplot as plt

a = 1.0
sigma = 0

time = np.arange(-2.1, 2.1, 0.1)

h_list = [1.0 / a if np.abs(t) <= a / 2.0 else 0 for t in time]

plt.figure(figsize=(10, 5))

for i in range(10):
    sigma += 0.1
    gaussian_deriv = []
    for t in time:
        gaussian_deriv.append(
            -t
            * np.exp(((-(t ** 2)) / (2 * sigma ** 2)))
            / (np.sqrt(2 * np.pi) * sigma ** 3)
        )

    convolved = np.convolve(h_list, gaussian_deriv)

    plt.plot(convolved)

plt.title("Convolution of Box Function with Gaussian derivative")
plt.show()
