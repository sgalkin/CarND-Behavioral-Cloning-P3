# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles
  from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without
  leaving the road
* Summarize the results with a written report
---

## Rubric
1. The submission includes a model.py file, drive.py, model.h5 a writeup report
and video.mp4.
2. The model provided can be used to successfully operate the simulation.
3. --_The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory._-- The model.py code is clearly organized and comments are included where needed.
4. --_The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model._--
5. --_Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting._--
6. --_Learning rate parameters are chosen with explanation, or an Adam optimizer is used._--
7. --_Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track)._--
8. The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.
9. The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.
10. The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included.
11. No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).
---

## Project structure

The project includes the following files:
* `model.py` - containing the script to create and train the model
* `drive.py` - for driving the car in autonomous mode
* `model.h5` - containing a trained convolution neural network
* `registry.py` - containing helper class for handling driving log
* `reader.py` - banch of helper classes to iterate through data points
* `repository.py` - helper class the provides git-like filesystem layout for
  huge directories
* `README.md` - project quick overview and summarizing the results

## Usage
Using the UdaCity provided simulator [1] and my `drive.py` file, the car can be
driven autonomously around the track by executing

```sh
python drive.py model.h5
```

## Model Architecture and Training Strategy

### Solution Design Approach

The overall strategy for deriving a model architecture was to:
*  Collect relevant training data
*  Analyze collected data to better understand it
*  Preprocess and normalize collected data
*  Use appropriate data augmentation techniques
*  Tune network parameters in order to achieve the best performance
*  Validate model on the simulator

My first step was to use a convolution neural network model similar to the one
described in [2]. I thought this model might be appropriate because the original
work was focused on the similar task.

In order to gauge how well the model was working, I split my image and steering
angle data into a training, validation, and test sets. See data analyze section
for details.

_I found that my first model had a low mean squared error on the training set but
a high mean squared error on the validation set. This implied that the model was
overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving
around track one. There were a few spots where the vehicle fell off the track...
to improve the driving behavior in these cases, I ...._

At the end of the process, the vehicle is able to drive autonomously around
the track without leaving the road.

### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a
combination of center lane driving in forward and backward direction,
recovering from the left and right sides of the road.

#### Creation of the Training Set

To capture good driving behavior, I first recorded three laps on track one
(lake) trying to keep center lane driving. Here is an example image of center
lane driving:

![Central lance driving](./images/central.gif)

I then recorded the vehicle recovering from the right side and left sides of
the road back to center so that the vehicle would learn how to fix position on
a road. This animation shows what a recovery looks like starting from left side:

![Recovery driving](./images/recovery.gif)

Next I recorded three laps on track one of center lane driving in opposite
direction.

Then I repeated this process on track two (jungle) and track three (castle,
found in previous version of simulator) in order to get more data points.

After the collection process, I had _53688_ data points. The data points were
divided to training, validation and test sets (randomly shuffled).
The actual numbers are

| Name           | %   | # of points |
|----------------|-----|-------------|
| Training set   | 70% | _37581_     |
| Validation set | 12% | _6442_      |
| Test set       | 18% | _9665_      |

#### Quality of the Data

Raw data distribution presented on figure below. Note logarithmic scale.

![Raw data distribution](./images/original.png)

'Straight driving' pattern was over represented in the data, which may lead to
biased network which unable to steer. In order to prevent high 'go straight'
bias I filtered out 80% (the parameter chosen experimentally ) of data points
with _0_ steering angle.
The distribution changed to the following (still over represented, but better).

![Filtered data distribution](./images/filtered.80.png)

#### Augmentation of the Data

Bellow I described augmentation techniques used in the project.

##### Image flipping

In order to obtain more data points in a safe way images were flipped around
y-axis, steering angle changed sign. The technique makes sense since it
produces valid (which may be captured by camera) image and appropriate steering
angle. This technique might lead to model overfitting since representation of
some features doubled.

![Flipped Images](./images/flipped.png)

##### Images from Side Cameras

The simulator provides two additional (left and right) images for each data
point. This images may be used in training process. The biggest problem is to
find appropriate steering angle correction. Also one have to keep in mind the
correction angle will be over represented in the data (since 'go straight') is
over represented.
After several experiments with static and random correction angles, I choose
correction angle to be _normally distributed with mu=0.15, sigma=0.03_.
Randomization in my opinion should reduce chance of overfitting.

##### Random Brightness and Contrast

Another possible augmentation technique - random brightness and/or contrast
adjustment. Using this technique allows model to learn how to deal with
different light conditions (sun sets, mid-days, shadows). Randomization reduces
chance of overfitting.

I used brightness adjustment with uniform distributed in range _[-20, 20]_.

![Brightness](./images/brightness.png)

##### Rotation (Roll)

Rotation could be used to produce more data. For example slightly rotated images
could help the network to learn how to drive in hilly environment
(ie. jungle track).

##### Steering Angle Randomization

The thing a believe helps a lot with fighting against overfitting - steering
angle randomization. The nature is continuous. In most cases small
change of value will not change final behavior, but will allow to smooth
data distribution.
I my experiments I used normally distributed randomization with
_mu=originalAngle and sigma=0.003_

##### Pipeline

Training data grooming pipeline
1. Angle adjustment for images from left and right cameras
2. Flipping images
3. Random brightness adjustment
4. Generating N (_2_ in my case) additional images using rotation
5. Angle randomization

Data distribution after pipeline:

![After Pipeline](./images/pipeline.png)

#### Preprocessing

Each image pass the following preprocessing steps before feeding into the
network (some steps implemented as lamda layers)

##### Colorspace Conversion

All images converted to YUV colorspace as proposed in [2].

##### Crop

Since top of each images shows sky and landscape (which is not much useful
for the task) and bottom shows parts of the car. Obvious decision is to crop
the image in order to reduce size (less to data to process) and focus network
on useful details.

![Cropped](./images/cropped.png)

##### Normalization

Each channel of an image normalized in order to have value in range _[-1, 1]_.

### Model

The final model architecture (`model.py` lines !TBD!) consisted of a convolution
neural network with the following layers and layer sizes

| Layer    | Input       | Output     | Kernel | Filters | Stride | Activation |
|----------|-------------|------------|--------|---------|--------|------------|
| Cropping | (160, 320, 3) | (80, 320, 3) |        |         |        |        |
| Lambda   | (80, 320, 3)  | (80, 320, 3) |        |         |        |        |
| Convolution  | (80, 320, 3)  | (38, 158, 24) | (5, 5) | 24 | (2, 2) | ReLU |
| Convolution  | (38, 158, 24) | (17, 77, 36)  | (5, 5) | 36 | (2, 2) | ReLU |
| Convolution  | (17, 77, 36)  | (7, 37, 48)   | (5, 5) | 48 | (2, 2) | ReLU |  
| Convolution  | (7, 37, 48)   | (5, 35, 64)   | (3, 3) | 64 | (1, 1) | ReLU |
| Convolution  | (5, 35, 64)   | (3, 33, 64)   | (3, 3) | 64 | (1, 1) | ReLU |
| MaxPooling   | (3, 33, 64)   | (1, 31, 64)   | (3, 3) |    | (1, 1) |      |
| Flatten      | (1, 31, 64)   | 1984          |        |    |        |      |
| Dropout(0.5) | 1984          | 1984          | | | | |
| Dense        | 1984          | 100           | | | | ReLU   |
| Dense        | 100           | 50            | | | | ReLU   |
| Dense        | 50            | 10            | | | | ReLU   |
| Dense        | 10            | 1             | | | | Linear |

![Model](./images/model.png)

All layers except the output use ReLU activation to introduce nonlinearity.
The model includes RELU layers to introduce nonlinearity.
Model uses Keras cropping and lambda layers in order to prepare (crop and
normalize) input data. To reduce overfitting the mode uses dropout layer.

### Training Process
I used this training data for training the model. The validation set helped
determine if the model was over or under fitting. Finally test data helped to
determine real model performance. I used an Adam optimizer so that manually
training the learning rate wasn't necessary.

The model was tested by running it through the
simulator and ensuring that the vehicle could stay on the track.

## Potential Improvements
* Tensorflow train data generation
* Multiprocessing for train data generation

## Data Sets

## References
1. Udacity's Self-Driving Car Simulator
   <https://github.com/udacity/self-driving-car-sim>
2. Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp,
   B., Goyal, P., Jackel, L. D., Monfort, M., Muller, U., Zhang, J.,
   et al. (2016). End to end learning for self-driving cars.
   arXiv preprint arXiv:1604.07316.
3. Clevert, D.-A., Unterthiner, T., Hochreiter S. (2015). Fast and
   accurate deep network learning by exponential linear units (elus).
   arXiv preprint arXiv:1511.07289, 2015.
