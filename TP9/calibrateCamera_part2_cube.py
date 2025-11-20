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


calib_file = "TP9/calib.npz"
if not os.path.exists(calib_file):
    print("ERREUR : Le fichier de calibration n'existe pas :", calib_file)
    exit(0)
data = np.load(calib_file)
mtx = data["mtx"]
dist = data["dist"]
print("Paramètres chargés.")



axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(np.uint16(imgpts[i])), tuple(np.uint16(imgpts[j])),(255),3)
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

while True:

    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

        #===========================exemple 1============================
        # projeter nimporte quel objet sur image
        # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        #=========================================================================
        
        R, _ = cv2.Rodrigues(rvecs) # vecteur --> matrice
        t = tvecs.reshape(3,1) # t c'est la translation
        K = mtx  # mtx c'est k qui contient fx, fy et positions
        X = axis.T                   # shape (3, N) axe coordonnées monde
        
        
        #===========================exemple 2=======================================
        X_cam = R @ X + t            # → coords caméra ( changer de repere seulement pas de pixel)
        x = K @ X_cam                # → coords homogènes (avoir pixel)
        #================================exemple 3=========================
        # matrice world to camera ( contient Rotation R matrice, translation T vecteur et 0 0 0 1)
        
        W2C = np.hstack((R, t))   # shape 3×4
        N = X.shape[1]
        X_h = np.vstack((X, np.ones((1, N))))   # (4,N)
        X_cam = W2C @ X_h         # (3,N)
        x = mtx @ X_cam           # (3,N)
        
        
        #=================================================================
        
        u = x[0, :] / x[2, :] # on divise sur z pour avoir coordonnées images (pixel)
        v = x[1, :] / x[2, :]
        imgpts = np.vstack((u, v)).T
        
        #==================================================================
        img = draw(img,corners2,imgpts)

    cv2.imshow('TP9/fille.jpeg',img)
    if cv2.waitKey(10)&0xFF == ord('0') :
        break
 
cv2.destroyAllWindows()
cap.release()