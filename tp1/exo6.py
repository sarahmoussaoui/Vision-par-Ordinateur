from random import randrange
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Creer image sombre max 13

# def CreatImage(h,w):
#     img=np.zeros((h,w),np.uint8)
#     randPointY, randpointX = randrange(h),randrange(w)
#     img[x,y]= 

img = cv2.imread('images/fille.jpeg', cv2.IMREAD_GRAYSCALE)
img[:]=img[:]/2
imgNorm = np.zeros((img.shape), np.uint8)
h, w = img.shape
min = 130
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

# Utiliser calcHist au lieu de boucles pour éviter l'overflow
hist_avant = cv2.calcHist([img],[0],None,[256], [0,256])

hist_apres= cv2.calcHist([imgNorm],[0],None,[256], [0,256])

plt.figure()
plt.title("image normalisé")
plt.xlabel("NG")
plt.ylabel("NB_pixels")
plt.plot(hist_avant.ravel())
plt.plot(hist_apres.ravel())
plt.xlim([0,255])
plt.show()

voisins=3

def filtreMoyenne(img):
        h,w = img.shape
        imgMoy= np.zeros(img.shape,img.dtype)
        for y in range(h):
                for x in range(w):
                        if y < voisins/2 or y > (h -voisins/2) or x < voisins/2 or x > (w-voisins/2) :
                              imgMoy[y,x] = img[y,x]  
                        else:
                                m = int(voisins/2) #marge bcs imgMoy[y,x] is in the middle
                                imgvoisins = img[y-m:y+m+1,x-m:x+m+1]
                                moy = 0
                                for yv in range(imgvoisins.shape[0]):
                                        for xv in range(imgvoisins.shape[1]):
                                                moy += int(imgvoisins[yv, xv])
                                moy /= voisins*voisins  # moy = sum / nbre de pixels 3*3
                                imgMoy[y,x] = moy
                                # imgMoy[y,x] = np.mean(imgvoisins)
        return imgMoy


imgMoy = filtreMoyenne(imgNorm)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1); plt.imshow(imgNorm, cmap='gray'); plt.title("Image bruitée"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(imgMoy, cmap='gray'); plt.title("Filtrage moyen"); plt.axis("off")
plt.show()


# Utiliser calcHist au lieu de boucles pour éviter l'overflow
hist_avant = cv2.calcHist([imgNorm],[0],None,[256], [0,256])

hist_apres= cv2.calcHist([imgMoy],[0],None,[256], [0,256])

plt.figure()
plt.title("image normalisé avec filtres")
plt.xlabel("NG")
plt.ylabel("NB_pixels")
plt.plot(hist_avant.ravel())
plt.plot(hist_apres.ravel())
plt.xlim([0,255])
plt.show()