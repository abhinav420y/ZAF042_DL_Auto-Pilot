## Self-Driving Car Steering Prediction
This project aims to train a deep neural network model to predict steering angles for a self-driving car using a dataset of images captured from a camera mounted on the car.

## Dataset
The dataset consists of a collection of images captured from the front-facing camera of a self-driving car. Each image is labeled with a steering angle value corresponding to the direction the car should steer while traveling. The dataset contains images captured while driving on both straight and curved roads, as well as images captured in different lighting conditions and weather.

## Model
We use a deep neural network model based on the NVIDIA architecture for training and prediction. The model takes in an image as input and outputs a steering angle value.

## Preprocessing
Before training the model, we preprocess the images by performing the following steps:

- Crop the top and bottom of the image to remove unnecessary features
- Convert the image to YUV color space
- Apply Gaussian blur
- Resize the image to a smaller size for easier processing
- Normalize the pixel values to be between 0 and 1

## Training
We train the model using a subset of the dataset, with 80% of the data used for training and 20% for validation. We use the Adam optimizer and mean squared error loss function. We also use early stopping to prevent overfitting.

## Results
The model achieves an accuracy of X% on the validation set, with an average mean squared error of Y. We also evaluate the model on a separate test set and achieve an accuracy of Z% with an average mean squared error of W.

## Usage
To use the model for prediction, you can load the saved model weights and use the predict function to get steering angle values for new images.

## Requirements
- Python 3
- Keras
- TensorFlow
- NumPy
- OpenCV

## Authors
KLAU-S