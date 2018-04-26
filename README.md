
# Identify Hand-Written Digits

## Introduction

>In this project, we build a model to correctly identify hand-written digits. We use MNIST dataset as our data. Our model will also be able to identify digits from zero to nine in any handwriting. We chose this problem because of its many real-world applications like toy touchpad that can identify writing digits. We did not spend too much time to do the data preprocessing since the dataset is already cleaned. In this project, we use Pytorch to build deep learning model. Moreover, we did a comparison of these two networks.
 
>The MNIST database of handwritten digit from “0” to “9” available on kaggle website.  All of the digits written by Census Bureau employees and high school students. This data set has 60,000 patterns for training sets and 10,000 patterns for the test set. The size of each image is 28 pixels by 28 pixels. Usually, it be used to judge the accuracy of deep learning. 

## Data Source

>The dataset available on the Kaggle, the link as following: 
       
>https://www.kaggle.com/c/digit-recognizer/data 
 
>However, in this project, we load the data by using the torchvision library.

## Setup
* **Platform：** Pycharm Community( Download from following link:https://www.jetbrains.com/pycharm/download/#section=windows)
* **Dependencies:** Python 3.6
* **Modules：**        
>* Torch             
>* Torchvision  
>* Matplotlib 
>* Scikit-learn  
>* Numpy

# Project Files
>* **README.md:** Describing the project and how to run the code 
* **train.csv:** The training dataset that we used for this project
* **test.csv:** The testing dataset that we used for this project
* **Code_CNN_Final_ML2.py:** The Code of convolution neural network
* **Code_MLP_Final_ML2.py:** The Code of multilayer neural perceptron
* **Report_Final_ML2.pdf:** The report of our project 
* **ML2 - Final Project-Slice.pptx:** The presentation of our project

# Installation

**To run this notebook:**  
**1. Download the file in your Downloads folder**  
>For Windows Users:  
* 1).Download git from the following link: https://git-scm.com/download/win and install it by simple click "Next" command 
* 2).Go to your Desktop, and right click left-click the mouse, and type, then choose "Git Bush Here 
* 3).In the git, type 'cd ~/Downloads'
* 4).Then, type 'git clone https://github.com/evazhang720/Final-Group-8.git'
> For Mac Users:
* 1).Open the 'Spotlight Search'
* 2).Type the'terminal' in the 'Spotlight Search'
* 3).In the terminal, type 'cd Downloads'
* 4). Then, type 'git clone https://github.com/evazhang720/Final-Group-8.git''
 
**2. Open the file**
>* 1).Open the 'Pycharm CE'
* 2).Click 'File', then click 'Open', then click 'Download'. In this path, you will find a folder names Code, in the code, you will find Code_CNN_Final_ML2.py and Code_MLP_Final_ML2.py

# Goal 

>In this project, we will apply deep learning techniques to build the optimal neural network to solve problem that from real-life.

# Conclusion

>The accuracy of MLP network is rated as 98% and of the convolution network at 99%.  This shows that both networks are highly accurate in reading the hand written digit and with further tweaking they can perform at almost 100% accuracy.  So according to our testing both network are suitable for this project of reading hand written digits.
