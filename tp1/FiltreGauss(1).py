import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import uint8

#loading image
img = cv2.imread(r'd:\Vision\t.jpeg', cv2.IMREAD_GRAYSCALE)


sigma = 1.4
x = 0
y = 0
e = math.e

def gauss(x, y):
    part1 = 1 / (2 * math.pi * pow(sigma, 2))
    part2 = -((x * x) + (y * y)) / (2 * pow(sigma, 2))
    return part1 * math.pow(e, part2)


def gauss_filter(size=3, sigma=1.0):
    
    filter_mat = np.eye(size, dtype=float)
    somme = 0.0
    k = size // 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            val = gauss(i, j)
            filter_mat[i + k, j + k] = val
            somme += val
            
    # Normalisation
    filter_mat /= somme

    return filter_mat


# Exemple d'utilisation :
f3 = gauss_filter(3, 1.0)
f5 = gauss_filter(5, 1.4)

print("Filtre Gaussien 3x3 :\n", np.round(f3, 4))
print("\nFiltre Gaussien 5x5 :\n", np.round(f5, 4))