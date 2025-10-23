import cv2
import numpy as np

th= 0
type_th = 0
img = cv2.imread('tp4/fille.jpeg', cv2.IMREAD_GRAYSCALE)
print(img.shape)
# fonction pour calculer le gradient : but determiner les contours en effectuant des differences de niveaux de gris entre pixels voisins
#  une dexuxieme image qui contient les valeurs des gradients
#  le graient par rapport a x et y c'est la racine
# pour dire que c'est un contour on peut faire un seuillage sur l'image des gradients

def gradient(img):
    h, w = img.shape
    imgGrad = np.zeros((h, w), np.uint8) # contains zeros because it treats borders too
    imgx = np.zeros((h, w), np.uint8)
    imgy = np.zeros((h, w), np.uint8)
    for y in range(1, h-1):
        for x in range(1, w-1):
            gx = int(img[y, x+1]) - int(img[y, x-1])
            gy = int(img[y+1, x]) - int(img[y-1, x])
            grad = int((gx**2 + gy**2)**0.5)
            imgx[y, x] = gx
            imgy[y,x] = gy
            imgGrad[y, x] = grad
   
    return imgGrad, imgx, imgy

def gradient_sans_boucle(img):
  #  mettre gradient a 0 --> dupliquer les bords
  # img with a border duplicated on the right
  h, w = img.shape()
  img_padded_right = np.zeros((h, w+1), np.uint8)
  img_padded_right[:,-2] = img
  img_padded_right[:,-1] = img[:,-1]

  img_grad = np.zeros((h, w), np.uint8)
  img_grad= img_padded_right - img
  

   

def afficher():
  imgRes = np.zeros_like(img)
  cv2.threshold(img, th, 255, type_th, imgRes)
  cv2.imshow('img', imgRes)


def change_th(x):
  global th
  th = x
  afficher()

def change_type(x):
  global type_th
  type_th = x
  afficher()



afficher()
gradient_img, imgx , imgy = gradient(img)
cv2.imshow('gradient', gradient_img) 
cv2.imshow('imgx', imgx)  
cv2.imshow('imgy', imgy)  

cv2.createTrackbar('thresh', 'img', 0, 255, change_th)
cv2.createTrackbar('type', 'img', 0, 4, change_type)
cv2.waitKey(0)
cv2.destroyAllWindows()