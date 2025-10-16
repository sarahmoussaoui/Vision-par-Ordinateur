import cv2
import numpy as np
import matplotlib.pyplot as plt

# from numpy.lib.shape_base import vsplit

img = cv2.imread('images/fille.jpg', cv2.IMREAD_GRAYSCALE)
img[:]=img[:]/2
imgNorm = np.zeros((img.shape), np.uint8)
h, w = img.shape
min = 255
max = 0

for y in range(h):
    for x in range(w):
        if img[y, x] > max:
            max = img[y, x]
        if img[y, x] < min:
            min = img[y, x]

for y in range(h):
    for x in range(w):
        imgNorm[y, x] = ((img[y, x] - min) / (max - min)) * 255.

cv2.imshow('image source', img)
cv2.imshow('image normal', imgNorm)
cv2.waitKey(0)
cv2.destroyAllWindows()

hist1 = np.zeros((256, 1), np.uint64)
for y in range(h):
    for x in range(w):
        hist1[img[y, x], 0] += 1

hist2 = cv2.calcHist([imgNorm], [0], None, [256], [0, 255])

plt.figure()
plt.title('Image normalisee')
plt.xlabel('GrayScale')
plt.ylabel('nbPixels')
plt.plot(hist2)
plt.plot(hist1)
plt.xlim([0, 255])
plt.show()