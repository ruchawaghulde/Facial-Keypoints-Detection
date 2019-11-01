# Facial-Keypoints-Detection

## By Lois, Siddharth and Rucha

EE-628-A Deep Learning: Final Project using PyTorch.
* The link :- https://www.kaggle.com/c/facial-keypoints-detection/overview.

## MOTIVATION:

Our reason for choosing this project is primarily based on our interest in applying deep learning to significant problems currently prevelant in the industry . Facial recognition is a very popular biometric technique these days. Various developments have already been observed in facial recognition technologies, but there is still a huge scope and a need for improvement. We are excited about applying the applications of Facial Key Points Detection as a building block in several applications such as tracking faces in images and videos, Analysing Facial Expressions, Detecting Dysmorphic Facial Signs for Medical Diagnosis and Biometrics/ Face Recognition. 

Detecting facial keypoints is a challenging problem given the variations in both facial features as well as image conditions. Facial features may differ according to size, position, pose and expression, while image qualtiy may vary with illumination and viewing angle. These variations, in combination with the necessity for highly accurate coordinate predictions (e.g. the exact corner of an eye) have lead us to believe that through this project we will be gaining insights about a deep and an interesting topic.

We specifically chose the Facial Keypoint Detection project because it will give us an opportunity to experiment with a wide variety of deep learning approaches, work with different neural networks and solve the problems associated with it.

## INTRODUCTION: 

In this project we are going to address the problem proposed above by creating a model that can detect the facial features from the image dataset. The main goal is to obtain the coordinates of eyes, eyebrows, nose and mouth in the picture. These coordinates are known as keypoints. In order to be more specific about the location and orientation of these keypoints, it will be necessary in some cases to assign more than one keypoint for each facial feature. This way, the face of the subject can be perfectly defined. For this dataset, our model will provide the following coordinates:

1)	**Eyes**: For both eyes the model will predict three coordinates corresponding to the center, inner and outer parts of the eyes.
2)	**Eyebrows**: For this feature the model will yield two coordinates corresponding to the inner and outer side for both of the eyebrows.
3)	**Nose**: In this case, one coordinate will be enough.
4)	**Mouth**: For the last feature, the model will give four coordinates, corresponding to the left, right, top and bottom part of the lip. This way the computer could actually read the mood of the subject.

In the past few years, advancements in Facial Key Points detection have been made by the application of *Deep Convolutional Neural Network* (DCNN). DCNNs have helped build state-of-the-art models for image recognition, recommender systems, natural language processing, etc. Our intention is to make use of these architectures to address this problem, trying to use different algorithms to study which are more suitable for the proposed task.
 

## DATA FILES:

The data files which we are going to use are as follows:

•	**training.csv**: list of training 7049 images. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels.

 ![train](https://user-images.githubusercontent.com/50252196/68026058-e20e9580-fc84-11e9-94d8-fc8106f5cc76.png)

•	**test.csv**: list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels

 
<img width="421" alt="test" src="https://user-images.githubusercontent.com/50252196/68026130-17b37e80-fc85-11e9-88ab-99d14aa3ce99.png">

•	**submissionFileFormat.csv**: list of 27124 keypoints to predict. Each row contains a RowId, ImageId, FeatureName, Location. FeatureName are "left_eye_center_x," "right_eyebrow_outer_end_y," etc. Location is what you need to predict. 

<img width="168" alt="submission" src="https://user-images.githubusercontent.com/50252196/68026212-6103ce00-fc85-11e9-9886-67c51d40145f.png">

## IMPLEMENTATION PLAN:

We will be using PyTorch to implement our project. Other libraries like numpy, pandas, matplotlib, etc. will be used based on our project requirements.
The implementation workflow will be as follows:
1)	**Data Pre-processing**: Mostly the datasets we get are not ready to be used for training since it requires certain amount of data pre-processing. We can perform data pre-processing to check if our dataset contains any missing values. 
We can use this process to separate features and labels from our dataset. The image is our feature and other values are labels that we have to predict later.
We can also reshape the images according to our requirements. 
So, this data pre-processing step will include loading and reading the dataset and making it ready for training.


2)	**Training**: For training, the algorithm which we are planning to use is *Convolutional Neural Network* (CNN). Other algorithms like *Adam* or *Stochastic Gradient Descent* will be used to minimize the loss function associated with this data.

3)	**Predictions**: After training our model using CNN, we can evaluate our predictions using the test dataset. This will let us know how our model will be training and decide on whether we should modify its structure or hyperparameters. This will also help us to find how our loss function behaves over time. With these initial observations, we will make changes to our model and decide the best architecture before we train for many epochs and create a final model. Then we use this final trained model to predict/ detect facial keypoints on the image.

4)	**Visualizing the predictions**: We can visualize our predicted outputs to see predicted facial keypoints on the given face images.


## APPLICATIONS:

Some of the applications of Facial Keypoints Detection are as follows:

• **Prevent Retail Crime**<br/>

Identifying the Facial key points can be used to instantly identify when known shoplifters, organized retail criminals or people with a history of fraud enter retail establishments. Photographs of individuals can be matched against large databases of criminals so that loss prevention and retail security professionals can be instantly notified when a shopper enters a store that prevents a threat. 

• **Find Missing People**<br/>

Facial key points prediction can be used to find missing children and victims of human trafficking. As long as missing individuals are added to a database, law enforcement can become alerted as soon as they are recognized by recognition model based on facial key points—be it an airport, retail store or other public space. In fact, 3000 missing children were discovered in just four days using face recognition in India!

• **Protect Law Enforcement**<br/>

Identifying the facial key points of criminals can help police officers instantly identify individuals in the field from a safe distance. This can help by giving them contextual data that tells them who they are dealing with and whether they need to proceed with caution. As an example, if a police officer pulls over a wanted murderer at a routine traffic stop, the officer would instantly know that the suspect may be armed and dangerous and could call for reinforcement.

• **Smarter Advertising**<br/>

Facial key point detection has the ability to make advertising more targeted by making educated guesses at people’s age and gender. It’s only a matter of time before face-recognition becomes an omni-present advertising technology. Advertisements can be based on detecting age, personality, origin and can be more focused towards these consumers. 

• **Security for Phones**<br/>

Unlocking Smart Phones based on making prediction and detection of facial key points can ensure faster deployment speeds as well as identifying minute differences between twins and people with identical facial features.

• **Deep Fake**<br/>

Tackling Deep Fake. Identifying the real image or video of a person from the fake ones developed by hackers.
