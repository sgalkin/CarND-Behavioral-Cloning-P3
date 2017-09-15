import cv2
import numpy as np

def wimage(image):
    return image.astype(np.int32)

def nimage(wimage):
    return np.minimum(255, np.maximum(0, wimage)).astype(np.uint8)
    
def contrast(wimage, factor):
    channel_mean = np.mean(np.mean(wimage, axis=0), axis=0)
    return wimage*factor + (1 - factor)*channel_mean

def brightness(wimage, factor):
    return wimage + factor

def rotate(image, angle, scale):
    rows, cols = image.shape[0:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    return cv2.warpAffine(image, M, (cols, rows))
