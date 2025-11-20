import cv2
import numpy as np
import os
 
CHECKERBOARD = (5,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints = [] 
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

cap = cv2.VideoCapture(0)
if not cap.isOpened :
    print("erreur ")
    exit(0)


calib_file = "calib.npz"
if not os.path.exists(calib_file):
    print("ERREUR : Le fichier de calibration n'existe pas :", calib_file)
    exit(0)
data = np.load(calib_file)
mtx = data["mtx"]
dist = data["dist"]
print("Paramètres chargés.")


axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
def draw(img, corners, imgpts):
    corner = tuple(np.uint16(corners[0]).ravel())
    img = cv2.line(img, corner, tuple(np.uint16(imgpts[0]).ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(np.uint16(imgpts[1]).ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(np.uint16(imgpts[2]).ravel()), (0,0,255), 5)
    return img

while True:

    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # recuperer les pixels
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

        # projection point monde sur pixel (image) --> l'objet qu'on veut afficher
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)

    cv2.imshow('img',img)
    if cv2.waitKey(10)&0xFF == ord('0') :
        break
 
cv2.destroyAllWindows()
cap.release()