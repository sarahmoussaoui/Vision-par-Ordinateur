import cv2
import numpy as np
 
CHECKERBOARD = (5,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints = [] 
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32) # repere monde
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)  # 3D 

 
cap = cv2.VideoCapture(0)
if not cap.isOpened :
    print("erreur ")
    exit(0)

while True:

    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
         
        imgpoints.append(corners2)
 
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        print(len(objpoints))
    cv2.imshow('img',img)
    if cv2.waitKey(100)&0xFF == ord('0') :
        break
 
cv2.destroyAllWindows()

 
h,w = img.shape[:2]

# rotation vecteur, translation vecteur
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez("TP9/calib.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("Calibration sauvegardée dans calib.npz :")

print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

cv2.destroyAllWindows()
cap.release()