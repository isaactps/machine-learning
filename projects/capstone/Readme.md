# Dog Breed Classifier Project in PyTorch
This is the Readme file for the Dog Breed Classifier Project that is my Capstone project in Udacity Machine Learning Engineer Nanodegree

It is implemented by using PyTorch library for Python version 3.6 under Amazon Sagemaker Notebook instance.

**The original Github repo from Udacity's where this project is based on is [here](https://github.com/udacity/deep-learning-v2-pytorch.git)**


## Project Overview

Welcome to my Capstone project in the Udacity Machine Learning Engineer Nanodegree! In this project, the objective is to build an algorithm for a Dog Identification App. Given an image of a dog, the algorithm will identify an estimate of the dog's breed.  However, if an image of a  human is provided, the algorithm will identify the resembling dog breed of the human. Also, a sample result from the output of the algorithm is as shown below.

[![Sample Output](https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_dog_output.png)](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-dog-classification/images/sample_dog_output.png)

Along with exploring state-of-the-art CNN models for classification and localization, some important design decisions about the user experience for the Dog Identification App will have to be made.  THe goal for completing the Capstone project is to understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  As each model has its strengths and weaknesses, therefore the task of engineering a real-world application often involves solving many problems that may not have a perfect answer.


## Project Instructions

### Setup Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Make sure the necessary Python packages have already installed according to the README in the program repository.
5. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dog_app.ipynb
	```

__NOTE:__ Currently some code has already been implemented to get the project going. However, additional functionality are required to be implemented in order to complete the project.

__NOTE:__ In the notebook, one will need to train CNNs in PyTorch framework.  If the CNN is taking too long to train, the best option is to use a GPU machine to perform the training of the CNNs.


-----

## My Implementations (2 Parts)

### Part1: Building a CNN model on from Scratch

I have built a CNN model from scratch to solve the problem initally. This model has 3 convolutional layers. Also, all convolutional layers have kernel size of 3 and stride of 1. The first conv layer (conv1) takes the 224*224 input image and the final conv layer (conv3) has a total of 128 kernel filters. ReLU activation function is used here. The MaxPooling2d layer of (2,2) is used to reduce the input size by 4. There are also two fully connected (FC) layers with the final FC layer producing a 133-dimensional output that corresponds to the 133 categories of dog breeds. Finally, a dropout layer of 0.20 is added to to the first fully connected layer so as to avoid over-fitting. 

(conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

activation: relu

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     

activation: relu

(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 

activation: relu

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) 

(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) 

(dropout): Dropout(p=0.2, inplace=False) 

(fc1): Linear(in_features=100352, out_features=512, bias=True) 

(dropout): Dropout(p=0.2, inplace=False) 

(fc2): Linear(in_features=512, out_features=133, bias=True) 

​The CNN model created from scratch has an accuracy score of 12.0% (103/836) on the test dataset after training for **25 epochs**. Thus, it does not meet the the project benchmark of 71.3%.


### Part 2: Refinement: CNN model built from Transfer Learning

The CNN created from scratch achieves an accuracy score of 12% which does not meet the project benchmark of 71.3%. Thus, a new CC model is built via transfer learning by using the VGG16 model that is pre-trained on ImageNet dataset. To perform the transfer learning of this model, the final final connected layer which is the classification layer will have to be replaced with a new classification layer that will output 133 classes instead of the original 1000 classes from ImageNet dataset. 


### Evaluation of Transfer Learning CNN Model

The CNN model created using transfer learning with VGG16 has an accuracy score of 83.0% (700/836) on test dataset after training for **25 epochs**. Besides the accuracy score of 83.0%, the model also has high precision score of 84.9% and recall score of 82.7%. Thus, this model meets the project benchmark of 71.3%.
