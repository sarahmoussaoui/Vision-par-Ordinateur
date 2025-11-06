import cv2
import numpy as np

sizeDelate = 1
sizeErode = 1
cv2.namedWindow("Erosion")
cv2.namedWindow("Dilataion")
img = cv2.imread("fille.jpeg", cv2.IMREAD_GRAYSCALE)
cv2.threshold(img, 130, 255, 0, img)

def delate_func():
    # kernel = np.ones((sizeDelate, sizeDelate), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sizeDelate * 2 + 1, sizeDelate * 2 + 1))
    img_delate = cv2.dilate(img, kernel, iterations=1)
    # img_delate = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Dilataion", img_delate)

def erode_func():
    # kernel = np.ones((sizeErode, sizeErode), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (sizeErode * 2 + 1, sizeErode * 2 + 1))
    img_erode = cv2.erode(img, kernel, iterations=1)
    cv2.imshow("Erosion", img_erode)

def changeEsize(x):
    global sizeErode
    sizeErode = x
    erode_func()

def changeDsize(x):
    global sizeDelate
    sizeDelate = x
    delate_func()

cv2.createTrackbar("Size Erode", "Erosion", 0, 21, changeEsize)
cv2.createTrackbar("Size Delate", "Dilataion", 0, 21, changeDsize)

erode_func()
delate_func()

cv2.imshow("image source", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
