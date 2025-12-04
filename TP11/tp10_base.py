import cv2
import numpy as np
lo=np.array([120, 30, 100])
hi=np.array([140, 70, 120])
def detect_inrange(image):
    points=[]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #mask=cv2.inRange(image, lo, hi)
    #-----------------------------------------------
    mask = (image >= lo) & (image <= hi)
    mask = mask.astype("uint8") * 255
    mask = mask[...,0] & mask[...,1] & mask[...,2]
    #-----------------------------------------------
    
    elements=cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    elements=sorted(elements, key=lambda x:cv2.contourArea(x), reverse=True)
    for element in elements:
        ((x, y), rayon)=cv2.minEnclosingCircle(element)
        points.append(np.array([int(x), int(y),int(rayon),int(
                cv2.contourArea(element))]))
    return image, mask, points
VideoCap=cv2.VideoCapture(0)
while(True):
    ret, frame=VideoCap.read()
    cv2.flip(frame,1,frame)
    image,mask,points = detect_inrange(frame)
    cv2.circle(frame, (100, 100), 20, (0, 255, 0), 5)
    print(image[100,100])
    if (len(points)>0):
        cv2.circle(frame, (points[0][0], points[0][1]), 10, 
                   (0, 0, 255), 2)
    if mask is not None :
        cv2.imshow("mask",mask)
    cv2.imshow('image', frame)
    if cv2.waitKey(10)&0xFF==ord('q'):
        break
VideoCap.release()
cv2.destroyAllWindows()








