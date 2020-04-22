# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/model_performance_overfit.png "Model Performance (with epoch=20)"
[image2]: ./results/model_performance.png "Model Performance (with epoch=5)"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `README.md` summarizing the results
* `clone.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `results/model.h5` containing a trained convolution neural network
* `results/runl.mp4` for the playback of autonomous drive based on `results/model.h5`

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py results/model.h5 results/runl
```

#### 3. Submission code is usable and readable

The `clone.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 48 and 64 (`clone.py` lines 106-112). The model includes RELU layers to introduce nonlinearity (code line 107 and 111), and the data is normalized in the model using a Keras lambda layer (code line 104). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (`clone.py` lines 115). The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 141-146). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track as shown in `results/runl.mp4`.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`clone.py` line 140).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of original center lane driving and flipped center lane driving images so that a ML model will not be biased by counterclockwise drives. With only original center lane driving images, an ego vehicle fell down to the pound during a right turn.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple ML model and gradually increase it's complexity so that I won't take overly complex approach.

My first step was to remove unnecessary information from input images. In this project, students are supposed to predict only steering angle so that ego vehicle won't leave it's road. Since I can predict steering angle by only looking at specific part of an image, I decided to crop input images and reduce channels by using only S channel of HSV color model.

Then I implement a simple convolutional neural network with 2 convolutional layers and only 1 full connected layer. Since it seems steering angle can be predicted with simple features, I thought 2 convolutional layers are enough for this task. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. To combat the overfitting, I modified training epochs from 20 to 5 so that mean squared error for training set show the same level of one for validation set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I found that ego falls into the pound and fails to turn right. To improve the driving behavior in these cases, I applied color filter to input images so that the pound will not be seen in a ML model, and added flipped images so that a ML model will not biased for counterclockwise drive, which contains more left turns than right.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. You can see how the ego vehicle drives the first course by watching `results/runl.mp4`.

#### 2. Final Model Architecture

The final model architecture (`clone.py` lines 96-123) consisted of a convolution neural network with the following layers and layer sizes.

|Layer                        |Output Shape        |      Param #     |
|:---------------------------:|:------------------:|:-----------------|
|InputLayer                   |(None, 20, 80, 1)   |      0           |
|Lambda                       |(None, 20, 80, 1)   |      0           |
|Conv2D with ReLU             |(None, 18, 78, 48)  |      480         |
|MaxPooling2D                 |(None, 9, 39, 48)   |      0           |
|Conv2D with ReLU             |(None, 7, 37, 64)   |      27712       |
|MaxPooling2D                 |(None, 3, 18, 64)   |      0           |
|Dropout (0.2)                |(None, 3, 18, 64)   |      0           |
|Flatten                      |(None, 3456)        |      0           |
|Dense                        |(None, 100)         |      345700      |
|Dence + Softmax              |(None, 1)           |      101         |

Lambda layer is used to normalize input images so that their mean values become 0, and avoid ML model to be sensitive to a large values.

#### 3. Creation of the Training Set & Training Process

I used dataset provided by Udacity, and didn't collect additional drive logs because I succeeded to let the ego vehicle finish the course 1 without leaving the road. However, I augmented the data set by flipping images so that ML model won't overfit to counterclockwise drive. To have better train, validation and test dataset, I randomly shuffled the data set and put 10% of the data into test set, 64% of the data into train set, 16% of the data into validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the following figure. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![Model overfits when epoch is 20][image1]
