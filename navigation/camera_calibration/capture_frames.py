import numpy as np
import cv2
import os
import shutil

cap = cv2.VideoCapture(1)

dir=os.path.dirname(__file__)+'/images/'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(dir)

index=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    if  key& 0xFF == ord('q'):
        break
    if key&0xFF ==ord(' '):#space
        cv2.imwrite(dir+str(index)+'.png',frame)
        index+=1



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()