import cv2
from scipy import signal, misc
import itertools
import numpy as np
from matplotlib import pyplot as plt

H = 1
sigma = 0

time = np.arange(-10.5, 10.5, 0.5)

h_list = [-H / 2 if t < 0 else H / 2 for t in time]

plt.figure(figsize=(10, 5))

for i in range(10):
    sigma += 0.5
    gaussian_deriv = []
    for t in time:
        gaussian_deriv.append(
            -t
            * np.exp(((-(t ** 2)) / (2 * sigma ** 2)))
            / (np.sqrt(2 * np.pi) * sigma ** 3)
        )
    
    convolved = np.convolve(h_list, gaussian_deriv)

    plt.plot(convolved[30:55])

plt.title("Convolution of Step Edge with Gaussian derivative")
plt.show()
    
