import cv2

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
