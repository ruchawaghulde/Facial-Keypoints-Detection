# Facial-Keypoints-Detection 
(A Google Collab Project)

## By Luis, Siddharth and Rucha

EE-628-A Deep Learning: Final Project using PyTorch.
* The link :- https://www.kaggle.com/c/facial-keypoints-detection/overview.

## MOTIVATION:

Our reason for choosing this project is primarily based on our interest in applying deep learning to significant problems currently prevelant in the industry . Facial recognition is a very popular biometric technique these days. Various developments have already been observed in facial recognition technologies, but there is still a huge scope and a need for improvement. We are excited about applying the applications of Facial Key Points Detection as a building block in several applications such as tracking faces in images and videos, Analysing Facial Expressions, Detecting Dysmorphic Facial Signs for Medical Diagnosis and Biometrics/ Face Recognition. 

Detecting facial keypoints is a challenging problem given the variations in both facial features as well as image conditions. Facial features may differ according to size, position, pose and expression, while image qualtiy may vary with illumination and viewing angle. These variations, in combination with the necessity for highly accurate coordinate predictions (e.g. the exact corner of an eye) have lead us to believe that through this project we will be gaining insights about a deep and an interesting topic.

We specifically chose the Facial Keypoint Detection project because it will give us an opportunity to experiment with a wide variety of deep learning approaches, work with different neural networks and solve the problems associated with it.

## INTRODUCTION: 

In this project we have addressed the problem proposed above by creating a model that can detect the facial features from the image dataset. The main goal is to obtain the coordinates of eyes, eyebrows, nose and mouth in the picture. These coordinates are known as keypoints. In order to be more specific about the location and orientation of these keypoints, it will be necessary in some cases to assign more than one keypoint for each facial feature. This way, the face of the subject can be perfectly defined. For this dataset, our model will provide the following keypoints:

1)	**Eyes**: For both eyes the model will predict three coordinates corresponding to the center, inner and outer parts of the eyes.
2)	**Eyebrows**: For this feature the model will yield two coordinates corresponding to the inner and outer side for both of the eyebrows.
3)	**Nose**: In this case, one coordinate will be enough.
4)	**Mouth**: For the last feature, the model will give four coordinates, corresponding to the left, right, top and bottom part of the lip. This way the computer could actually read the mood of the subject.

In the past few years, advancements in Facial Key Points detection have been made by the application of *Deep Convolutional Neural Network* (DCNN). DCNNs have helped build state-of-the-art models for image recognition, recommender systems, natural language processing, etc. Our intention is to make use of these architectures to address this problem, trying to use different algorithms to study which are more suitable for the proposed task.
 

## DATA FILES:

The data files which we have used are as follows:

•	**training.csv**: list of training 7049 images. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels.

 ![train](https://user-images.githubusercontent.com/50252196/68026058-e20e9580-fc84-11e9-94d8-fc8106f5cc76.png)

•	**test.csv**: list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels

 
<img width="421" alt="test" src="https://user-images.githubusercontent.com/50252196/68026130-17b37e80-fc85-11e9-88ab-99d14aa3ce99.png">

•	**submissionFileFormat.csv**: list of 27124 keypoints to predict. Each row contains a RowId, ImageId, FeatureName, Location. FeatureName are "left_eye_center_x," "right_eyebrow_outer_end_y," etc. Location is what you need to predict. 

<img width="168" alt="submission" src="https://user-images.githubusercontent.com/50252196/68026212-6103ce00-fc85-11e9-9886-67c51d40145f.png">

## IMPLEMENTATION PLAN:

We have used PyTorch to implement our project along with libraries such as numpy, pandas, matplotlib, etc. will be used based on our project requirements.
The implementation workflow will be as follows:
1) **EDA and Feature Engineering**:

* Importing the data : Importing the json file from Kaggle which contain the username and the key for our Kaggle account.To link our Kaggle account to Google Collab, create a new API token on your Kaggle account.Creating a client by making a directory to host our Kaggle API token.After this step, use ‘kaggle competitions download -c facial-keypoints-detection’ API to import the csv and folders from the data source on Kaggle.Finally, decompress these folders to obtain our csv files.

 * Data Pre-processing:Using Pandas, create data frames for our csv files and extract features which can be used in our analysis.Since our data contains some missing values, this step involves calculating, visualizing and replacing those missing values for every feature.
 
 <p align = "center"><img width="600" img height="600" src="https://github.com/siddh30/Facial-Keypoints-Detection/blob/master/Observation%20Images/Missing%20Data.png"></p>
 
 We have also used heat maps by Seaborn for finding the correlation between features.
 
 <p align = "center"><img width="800" img height="600" src="https://github.com/siddh30/Facial-Keypoints-Detection/blob/master/Observation%20Images/HeatMaps%20for%20Correlation%20of%20features.png"></p>
 
Then we spilt the training into keypoints and images. Each row of keypoints data contains the (x, y) coordinates for 15 keypoints, while that of images data contains the row-ordered list of pixels.

* Visualizing the input image: Creating a numpy array of the pixel values in the image column of our training dataset.
Using matplotlib to plot the image from these pixel values.
Using features such as left_eye_center_x, nose_tip_x, etc to plot keypoints on face images.
Formulating a gaussian function to create heatmaps of these facial keypoints.

<p align = "center"><img width="700" img height="200" src="https://github.com/siddh30/Facial-Keypoints-Detection/blob/master/Observation%20Images/sample%2Bkeypoints%2Bheatmaps.png"></p>


2)	**Training**: For training, the algorithms which we have used are **CNN** and **LeNet**. We have chosen ** RELU** to be our Activation function and our Optimizer function is **Adam** used to minimize the loss function associated with this data.

3)	**Predictions**: After training our model using CNN and LeNet, we have evaluated our predictions using the test dataset helping us find how our loss function behaves over time. With these initial observations. We have also used used these models to predict keypoints on images from the internet.

4)	**Visualizing the predictions**: We have finally visualized our predicted outputs to see predicted facial keypoints on the given face images.

<p align = "center"><img width="600" img height="600" src="https://github.com/siddh30/Facial-Keypoints-Detection/blob/master/Observation%20Images/validation%20images%20real%20and%20predicted%20(1).png"></p>

Extending further with this implementation, we tried to take our own input image which is not necessarily a face image. We took an input image which contains a group of people. Haar cascade classifier is used for face detection in this image as shown in figure. Now we have detected the face images from our input image, we resized the face images so that it would fit our project requirements. 

<p align = "center"><img width="600" img height="400" src="https://github.com/siddh30/Facial-Keypoints-Detection/blob/master/Observation%20Images/original%2Bface%20boxes.png"></p>

Finally, we have applied our trained CNN and ResNet model to these detected face images. We successfully achieved facial keypoints detection of these images as shown in figure. So now we can use any image and locate keypoints on it.

<p align = "center"><img width="800" img height="100" src="https://github.com/siddh30/Facial-Keypoints-Detection/blob/master/Observation%20Images/faces%2Bkeypoints.JPG"></p>

## APPLICATIONS:

Some of the applications of Facial Keypoints Detection are as follows:

• **Prevent Retail Crime**<br/>


• **Find Missing People**<br/>


• **Protect Law Enforcement**<br/>



• **Smarter Advertising**<br/>


• **Security for Phones**<br/>


• **Deep Fake**<br/>
