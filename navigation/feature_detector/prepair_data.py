import cv2
import json
import config
import numpy

def prepair(image_fn,txt_fn,scale,alpha):
    image=cv2.imread(image_fn)
    mask=numpy.zeros((config.detector_input_W,config.detector_input_W,1))

    with open(txt_fn,'r+') as f:
        description=json.load(f)

    x=description['x']*config.detector_input_W/image.shape[1]
    y = description['y'] *config.detector_input_W / image.shape[0]

    image = cv2.resize(image, (config.detector_input_W, config.detector_input_W))/255.0
    mask = cv2.circle(mask, (int(x), int(y)), 20, 0.5, thickness=-1)
    mask=cv2.circle(mask,(int(x),int(y)),10,1.0,thickness=-1)
    image=rotate_and_scale_Image(image,(int(x),int(y)),scale,alpha)
    mask = rotate_and_scale_Image(mask, (int(x), int(y)), scale, alpha)
    mask=cv2.resize(mask,(int(image.shape[0]/8),int(image.shape[1]/8)))
    return image,mask


def rotate_and_scale_Image(image,point, scale,angle):

  rot_mat = cv2.getRotationMatrix2D(point, angle, scale)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

if __name__=='__main__':
    image,mask=prepair('images/0_0.png','images/0_0.txt',0.7,20.0)
    cv2.imshow('1',image)
    cv2.imshow('2', mask )
    cv2.waitKey()
