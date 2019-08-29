# Humpback-Whale-Identification-

Abstract
===
In this task, we need to build an algorithm to identify individual whales through their tails in images. We’ll use a database of over 25,000 images to match each whale species correctly by analyzing the picture of each whale’s humpback. Therefore, how to deal with huge quantities of databases and which model is suitable for the problem are our primary goals. Because our team members are familiar with Python, Jupyter notebook and the kernel system are our main platforms for this project. 
Once classifying the images from training sets, we use the method named ‘hash table’ to find duplicate or similar images. If the number of pictures in one class is smaller than 2, the datasets cannot form picture pairs to train the testing databases.  When selecting the picture pairs, duplicate, lower resolution images should be deleted.  And then these images’ color is translated into black and white (just one channel). The affine transformation is applied at the same time. 
In this project, our group uses two popular neural network models to identity the whales in test sets:

1. Squeeze-and-Excitation Network (SeNet) of which result does not meet our primary requirements. The accuracy of the submission file is about 37%. 
2. Siamese Neural Network, which is composed of Convolutional Neural Network (CNN), which includes six blocks, and a custom network mostly like (Residual Neural Network) ResNet. Although we are not satisfied with the scores at first, the final accuracy is more than 82% after four times improvements.

As a result, the highest accuracy is about 82% using siamese neural network combined with CNN and ResNet. We will get better results by adding up the number of epochs in the future. However, the time complexity of this algorithm is O (n^3), which costs us too much time to run more epochs. So the further step is to improve the model and find the best learning rate, optimizer and activate function to get better predictions.    

Contribution
===
Xinru Zhang is responsible to the data-preprocess part, while Zhenbo Liu and Mengjie Min take charge of SENet model and Siamese Neural Network model, respectively. As to improvements and the final report script, all of us work together.


