from sklearn.utils import shuffle
import cv2
import numpy as np

import functools
import itertools

'''This method is a heart of data generation used in the project. 
It reads input and apply piplined generators against it.

Parameters:
  * reader - reader object for input
  * repo - image repository
  * X_pipeline - tuple of piplince specs in a form - 
    (
      (index1, (transformation11, transformation12, ...)),
      (index2, (transformation21, transformation22, ...))
    )
  * y - measurment

Generate:
   Flatten list of images with given transformation applied
''' 
def generator(reader, repo, X_pipeline, y):
    def read_data(repo, name, s):
        yield cv2.imread(repo.resolve(name)), float(s)
    
    X, p = zip(*X_pipeline)
    return (generated
            for inputs in reader.read(*X, y) #TODO replace with generator
            for f, x in zip(p, inputs[:-1])
            for generated in 
                functools.reduce(lambda gen, fn: (generated 
                                                  for args in gen 
                                                  for generated in fn(*args)), 
                                 (read_data,) + f, 
                                 ((repo, x, inputs[-1]) 
                                  for _ in range(1))))

'''Combines single images into batches of given size, partial batch discarded'''
def batch_generator(gen, batch_size):
    while True:
        batch = tuple(np.array(x) for x in zip(*itertools.islice(gen, batch_size)))
        if len(batch) == 0 or len(batch[0]) != batch_size: 
            return
        yield batch

'''Shuffles batch. It is a good idea to shuffle the batch since neighbour images are related'''
def shuffle_batch(gen):
    return (shuffle(*batch) for batch in gen)

'''Generate infinite sequence of batches, required by Keras generator_fit'''
def infinite_generator(generate, shuffle):
    while True:
        shuffle()
        for v in generate():
            yield v
