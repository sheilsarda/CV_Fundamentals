import cv2
from scipy import signal, misc
import itertools
import numpy as np
from matplotlib import pyplot as plt

v = np.array([-2, -1, 0, 1, 2])
gaussian_1 = np.zeros((5, 5))
gaussian_2 = np.zeros((5, 5))

sigma_1 = 1
sigma_2 = 0.1

coords = itertools.product(v, v)
for (y, x) in coords:
    gaussian_1[y, x] = (
        1.0
        / (np.sqrt(2 * np.pi) * sigma_1)
        * np.exp(-(v[x] ** 2 + v[y] ** 2) / (2 * sigma_1 ** 2))
    )

    gaussian_2[y, x] = (
        1.0
        / (np.sqrt(2 * np.pi) * sigma_2)
        * np.exp(-(v[x] ** 2 + v[y] ** 2) / (2 * sigma_2 ** 2))
    )


img = cv2.imread("../imgs/statue.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
convolved_img_1 = signal.convolve2d(gaussian_1, gray)

plt.figure(figsize=(20, 20))
plt.imshow(convolved_img_1, cmap="Greys")
plt.show()

convolved_img_2 = signal.convolve2d(gaussian_2, gray)
plt.figure(figsize=(20, 20))
plt.imshow(convolved_img_2, cmap="Greys")
plt.show()
