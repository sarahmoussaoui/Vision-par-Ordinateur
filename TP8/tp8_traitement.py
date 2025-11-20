import cv2
import numpy as np 
import time
cap = cv2.VideoCapture("TP8/output_out3.avi")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

out = cv2.VideoWriter('TP8/output_out4.avi',fourcc,25,(frame_width, frame_height))

if not cap.isOpened():
  print("error capture")
  exit(0)
while(cap.isOpened()):
  ret, frame= cap.read()
  t_start = time.time()
  if not ret :
    print("error read frame")
    break
  
  frame= cv2.flip(frame,1)
  out.write(frame)
  cv2.imshow("image",frame)
  # if cv2.waitKey(50) & 0xFF == ord('q'):
  #   break
  while((1/(time.time()-t_start)) > 25): # 25 is time of sampling (25images per second)
    # cv2.waitKey(0)
    pass
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  print("time:", 1/(time.time()-t_start)) # temps d'échantillonage

out.release()
cap.release()
cv2.destroyAllWindows()
