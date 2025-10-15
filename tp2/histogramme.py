'''Histogramme des niveaux de gris '''
'''pour les images qui est sauvgarder sur 16 bits uint16... pour chque c'est l image est black dans on pred w * h si est elle toute noir et si h= 400 et w = 400 
donc on a 160000 pixel noir et pour les histogramme il ya 8 bits !
'''
import cv2
import numpy as np
img  = cv2.imread("download.jpg", cv2.IMREAD_GRAYSCALE)
#img[:] = img[:]//2
# img= cv2.

if img is None : 
    print("erreur de chargement")
    exit(0)
h,w = img.shape
min,max = 255,0
for y in range(h):
    for x in range(w) : 
        if (img[y,x] > max) : 
            max = img[y,x]
        if (img[y,x] < min) :
            min = img[y,x]

img_apres = np.zeros(img.shape,img.dtype)
for y in range (h) : 
    for x in range (w) : 
        img_apres = ((img[y,x] - min) * 255) / (max - min)

print("min:",min,"max:",max)
print("image avant : ", img)
print("image apres : ",img_apres)






# # import matplotlib.pyplot as plt 

# hist_avant = np.zeros((256,1), np.uint16)
# for y in range(h):
#     for x in range(w):
#         hist_avant[img[y,x]]+=1

# hist_apres = cv2.calcHist([img_apres],[0],None,[256],[0,255])

# plt.figure()
# plt.title("image normilzed")
# plt.xlabel("NG")
# plt.ylabel("NB Pixels")
# plt.plot(hist_avant)
# plt.plot(hist_apres)

# plt.xlim([0,255])

# plt.show()

cv2.waitKey(0)

