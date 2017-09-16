#!/usr/bin/env python

from registry import Registry
from repository import Repository
from reader import CycleReader, FilteredReader, LimitedReader
import generator
import pipeline
import model

import cv2
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

import argparse
import itertools
import os
import pickle


def prepare_data(path, reg, train_fraction, valid_fraction):
    def build_repository(registry):
        repo = Repository(os.path.join(registry.prefix, 'repo'))
        repo.build(registry.prefix,
                   itertools.chain(
                       *registry.read(Registry.CENTER, Registry.LEFT, Registry.RIGHT)))
        return repo
    
    def split_registry(registry, train_fraction, valid_fraction):
        registry.shuffle()
        valid, train = registry.split(train_fraction)
        test, valid = valid.split(valid_fraction / (1 - train_fraction))
        return train, valid, test

    registry = Registry(reg)
    repo = build_repository(registry)
    
    train, valid, test = split_registry(registry, train_fraction, valid_fraction)
    print('Train set:', len(train))
    print('Validation set:', len(valid))
    print('Test set:', len(test))
    
    test.store(os.path.splitext(path)[0] + '.test.csv')
    return train, valid, repo


def make_normalized_reader(reg, factor):
    def filter(x): # take non zero measurements and factor zero mesurments
        return float(x[-1]) != 0 or np.random.rand() < factor
    return FilteredReader(reg, filter)


def train(path, reg):
    def checkpoint(path):
        name, ext = os.path.splitext(path)
        pattern = os.path.join(name + '.checkpoint',
                               os.path.basename(name) + '.e{epoch:02d}.l{val_loss:.4f}' + ext)
        os.makedirs(os.path.dirname(pattern), exist_ok=True)
        return pattern

    def augmentation_factor(make_generator, reg, repo):
        return len(list(make_generator(LimitedReader(reg, 1), repo)))

    # Training Tunables. TODO: make configurable via arguments
    EPOCHS=30
    BATCH_SIZE=1280
    BOOST=1
    
    # Model Tunables
    INITIALIZER = 'truncated_normal'
    ACTIVATION = 'relu'
    OUT_ACTIVATION = 'linear'
    DROPOUT = None
    SPATIAL_DROPOUT = None

    # Compilation Tunables
    LOSS = 'mse'
    OPTIMIZER = 'adam'

    if os.path.exists(path):
        raise IOError('File {} already exists'.format(path))

    # prepare train and validation set, and build images repository
    # TODO: make fractions configurable
    train, valid, repo = prepare_data(path, reg, 0.7, 0.12)

    # take overpopulated 0 angle with given probability
    normalized_train = make_normalized_reader(train, 0.1)
    normalized_train_len = len(list(normalized_train.read(Registry.STEERING)))
    print('Normalized train set:', normalized_train_len)

    # evaluate augmentation factor for generator used in training
    augmentation_factor_train = augmentation_factor(
        lambda reg, repo: pipeline.train_generator(reg, repo, BOOST), train, repo)
    augmentation_factor_valid = augmentation_factor(pipeline.valid_generator, valid, repo)
    print('Augmenation factor train:', augmentation_factor_train)
    print('Augmenation factor valid:', augmentation_factor_valid)

    # create infitine batch generator for training data
    infinite_train_generator = generator.infinite_generator(
        lambda: generator.shuffle_batch(
            generator.batch_generator(
                pipeline.train_generator(normalized_train, repo, BOOST), 
                BATCH_SIZE)),
        lambda: normalized_train.shuffle())

    # create infinite batch generator for validataion data
    infinite_valid_generator = generator.batch_generator(
        pipeline.valid_generator(CycleReader(valid), repo), BATCH_SIZE)

    # evaluate input image shape
    input_shape=cv2.imread(repo.resolve(next(valid.read(Registry.CENTER)))).shape

    # build and compile the model
    m = model.model(input_shape=input_shape,
                    initializer=INITIALIZER,
                    activation=ACTIVATION,
                    out_activation=OUT_ACTIVATION,
                    dropout=DROPOUT)
    m.compile(loss=LOSS, optimizer=OPTIMIZER)

    # train the model
    history = m.fit_generator(
        epochs=EPOCHS,
        generator=infinite_train_generator,
        steps_per_epoch=(augmentation_factor_train*normalized_train_len)//BATCH_SIZE,
        validation_data=infinite_valid_generator,
        validation_steps=(augmentation_factor_valid*len(valid))//BATCH_SIZE,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, min_delta=0.0001),
            ModelCheckpoint(monitor='val_loss', filepath=checkpoint(path)),
        ],
    )

    # store model
    m.save(path)

    # store training history
    with open(os.path.splitext(path)[0] + '.history.p', 'wb') as h:
        pickle.dump(history.history, h)

        
def validate(path, reg):
    if not os.path.exists(path):
        raise IOError('File {} not found'.format(path))
    if not os.path.exists(reg):
        raise IOError('File {} not found'.format(reg))

    model = load_model(path)
    registry = Registry(reg)

    print('Loss: {:.4f}'.format(
        model.evaluate_generator(
            generator.batch_generator(
                pipeline.valid_generator(
                    CycleReader(registry), registry
                ), 1
            ), steps=len(registry))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training/validation')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-v',
                       action='store',
                       help='Validate model using given data')
    group.add_argument('-t',
                       action='store',
                       help='Train model using fraction of given data')

    args = parser.parse_args()
    if args.v:
        validate(args.model, args.v)
    else:
        train(args.model, args.t)
