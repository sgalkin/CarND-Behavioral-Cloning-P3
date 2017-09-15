import cv2
import numpy as np

import cvutils
import registry
from generator import generator
from registry import Registry

def uniform_rnd(a, b):
    return (b - a)*np.random.rand() + a

def normal_rnd(mu, sigma):
    return mu + sigma*np.random.randn()

def convert_colorspace(how):
    def converter(i, s, how):
        yield cv2.cvtColor(i, how), s
    return lambda i, s: converter(i, s, how)

def train_generator(reg, repo, boost):
    HOW = cv2.COLOR_BGR2YUV # source image colorspace
    NOISE_SIGMA = 0.002 # noise sigma for angle randomization
    CORRECTION_MU = 0.2 # angle correction for side images
    CORRECTION_SIGMA = 0.05 # angle correction sigma

    def angle_adjuster(mu, sigma):
        def adjuster(i, s, mu, sigma):
            yield i, s + normal_rnd(mu, sigma)
            
        return lambda i, s: adjuster(i, s, mu, sigma)
   
    def flip(i, s):
        yield i, s
        yield cv2.flip(i, 1), -s

    def augmentation(count, gen):
        def augment(i, s, count, gen):
            yield i, s
            for _ in range(count):
                for i, s in gen(i, s):
                    yield i, s
                    
        return lambda i, s: augment(i, s, count, gen)
 
    def random_bc(i, s):
        yield cvutils.nimage(
            cvutils.brightness(
                cvutils.wimage(i),
                uniform_rnd(-20, 20))), s

    def random_rotate(i, s):
        yield cvutils.rotate(i, uniform_rnd(-5., 5.), uniform_rnd(1.1, 1.2)), s
        
    center_processor = (Registry.CENTER, (
                                 flip,
                                 random_bc,
                                 convert_colorspace(HOW),
                                 augmentation(boost, random_rotate), 
                                 angle_adjuster(0, NOISE_SIGMA)
                                ))
    left_processor = (Registry.LEFT, (
                             angle_adjuster(CORRECTION_MU, CORRECTION_SIGMA),
                             flip,
                             random_bc,
                             convert_colorspace(HOW),
                             augmentation(boost, random_rotate), 
                             angle_adjuster(0, NOISE_SIGMA)
                            ))
    right_processor = (Registry.RIGHT, (
                               angle_adjuster(-CORRECTION_MU, CORRECTION_SIGMA),
                               flip,
                               random_bc,
                               convert_colorspace(HOW),
                               augmentation(boost, random_rotate), 
                               angle_adjuster(0, NOISE_SIGMA)
                              ))

    return generator(reg, repo, (
        center_processor, 
        left_processor, 
        right_processor
    ), Registry.STEERING)


def valid_generator(reg, repo):
    return generator(reg, repo, (
        (Registry.CENTER, (
            convert_colorspace(cv2.COLOR_BGR2YUV),
        ),),), Registry.STEERING)
