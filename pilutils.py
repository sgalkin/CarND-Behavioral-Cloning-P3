import math

from PIL import Image, ImageDraw
import numpy as np

def vector(image, angle, width=40, color=(0, 255, 0)):
    draw = ImageDraw.Draw(image)
    origin = np.array((image.size[0]/2, image.size[1]*0.9))
    direction = math.pi/180.*angle
    vec = width*np.array((math.sin(direction), -math.cos(direction)))

    draw.line((*origin, *(origin+vec)), width=3, fill=color) 
    return image

# Testing
def main():
    import time
    
    image = Image.open('foo/2017_09_10_16_44_55_136.jpg')
    print(image, image.size[0], image.size[1])
    image = vector(image, -45)
    image.show()
    time.sleep(1)

if __name__ == '__main__':
    main()
