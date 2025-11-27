import cv2
import numpy as np

img = cv2.imread("fille.jpeg", cv2.IMREAD_UNCHANGED)

if img is None:
    print("Erreur")
    exit()
else:
    h, w, c = img.shape
    imgRes = np.zeros((h, w), np.uint8)

    # B G R
    B = img[:, :, 0]   
    G = img[:, :, 1]
    R = img[:, :, 2]

print(B)