import numpy as np
import cv2
import os
import shutil
import config
import random
import json

cap = cv2.VideoCapture(1)

dir=os.path.dirname(__file__)+'/images/'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(dir)

ret, frame = cap.read()

# Display the resulting frame

if frame.shape[0] != config.cam_height or frame.shape[1] != config.cam_width:
    print('wrong frame size')
    exit()

feature_index=0
frame_index=0

# font
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

minX=config.cam_width/10
maxX=config.cam_width-minX
minY=config.cam_height/10
maxY=config.cam_height-minY
# Using cv2.putText() method

x = random.randint(int(minX), int(maxX))
y = random.randint(int(minY), int(maxY))

while(True):

    # Capture frame-by-frame
    ret,frame=cap.read()
    or_frame=frame.copy()

    cv2.line(frame,(x-20,y),(x+20,y),(0,0,255),thickness=2)
    cv2.line(frame, (x, y-20), (x , y+20),(0,0,255),thickness=2)
    frame = cv2.putText(frame, 'feature '+str(feature_index)+' frame_index '+str(frame_index), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('image', frame)

    key=cv2.waitKey(1)
    if  key& 0xFF == ord('q'):
        break
    if key&0xFF ==ord(' '):#space
        cv2.imwrite(dir+'/'+str(feature_index)+'_'+str(frame_index)+'.png',or_frame)
        with open(dir+'/'+str(feature_index)+'_'+str(frame_index)+'.txt','w+') as f:
            json.dump({'x':x,'y':y},f)

        x = random.randint(int(minX), int(maxX))
        y = random.randint(int(minY), int(maxY))

        frame_index+=1
        if frame_index==config.num_frames_for_features:
            frame_index=0
            feature_index+=1
        if feature_index==config.num_features:
            feature_index=0



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()