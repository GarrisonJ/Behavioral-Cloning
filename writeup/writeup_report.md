# **Behavioral Cloning Project** 

[//]: # (Image References)

[image1]: ./model_visualization.png "Model Visualization"
[image2]: ./center.jpg "Center"
[image3]: ./left.jpg "Left"
[image4]: ./right.jpg "Right"
[image5]: ./recovery.jpg "Recovery" 


## Project

This project was to design and train a neural network that could drive a simulated car around a track. This was for Udacity's self-driving car nanodegree.

## Model Architecture and Training Strategy

### Solution Design Approach

My network architecture was inspired by the architecture used by [Nvidia](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) to build their self-driving car. I simplified the model, so it has fewer parameters because the problem space is much smaller. The Nvidia architecture starts with a normalization layer, followed by five convolutional layers, followed by a flattening layer, followed by four dense layers, and finally, a single output for the vehicle control. My model is the same, except it only has two dense layers at the end of the network, and almost all of the layers have fewer parameters.

I also added dropout layers between the convolutional layers to help combat overfitting.

This architecture was successful in steering the car around the track without hitting the edges.

### Final Model Architecture

The final model architecture can be found in model.py. Also, here is a diagram:

![alt text][image1]

### Creation of the Training Set & Training Process

I created five datasets to train with:

1. Driving around track-1 twice
2. Driving around track-1 backwards twice
3. Driving around track-1 recovering from being off the road
4. Driving around track-2 twice
5. Driving around track-2 backwards twice


Here is am example of the left, center, and right cameras images from the car driving around track-1:

![alt text][image3]
![alt text][image2]
![alt text][image4]

And here is an example of the starting position of the car before recording its recovery, notice how the car is too far to the right and needs to drive left to recover:

![alt text][image5]

I used the images from all three cameras, the center, left, and right. I did this by offsetting the steering angle by 0.2 degrees for the left and right images. This would cause the algorithm to steer the car towards the center if the center camera seemed similar to the left or right camera. 

Using a lambda layer on the network, I preprocessed the images by normalizing the pixel values and cropping the top and bottom of the image. The top and bottom of the image didn't contain much useful information. 

The number of epochs was discovered empirically and I used an adam optimizer so that manually training the learning rate wasn't necessary.


