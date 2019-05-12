# TFJS-CustomImageClassification
Building Mobilenet Based Custom Image Classification Model on the Browser with Tensorflow.js and Angular

Tensorflow.js
There are several ways of fine tuning a deep learning model but doing this on the web browser with WebGL acceleration is something that we experienced not such a long time ago, with the introduction of Tensorflow.js. In this example, I will use Tensorflow.js together with Angular to build a Web App which trains a convolutional neural network to detect malaria infected cells with the help of Mobilenet and Kaggle dataset containing 27.558 infected and uninfected cell images.
Mobilenet as Base Model
As I mentioned above, I will use mobilenet as a base model for our custom image classifier. The pre-trained mobilenet model, which is tensorflow.js compatible, is relatively small, 20MB, and can be directly downloaded from the Google api storage folder. 
There will be 2 classes, which are uninfected and parasitized cells, we would like to classify with our custom model whereas the original mobilenet model is trained to classify 1000 different objects.
I will use all mobilenet layers other than last 5 layers, and add a dense layer with 2 units and softmax activation on top of this truncated model to make the modified model suitable for our classification task.
We will not train all layers, as this will require a lot of time and computational power. It is also not necessary for our case as we use a pre-trained model which has a lot of representations learned already. Instead, we will freeze majority of the layers and keep only last 2 ones trainable.
