# ppe_object_detection
make a plan of what you are going to do
Currently the pictorial data is in a format that can be accepted by YOLO.
The data is split into train and test

The first thing to do is to actually to use the test set to make predictions
This will give us an idea of the baseline performance of the model

Second thing is to finetune some layers of the model
of course we will be using the training images to finetune the layers of the model before training
Then to use the test set to see the MAP@0.5:0.95 score

We will begin by learning how to make use of the YOLO model
first for running predictions
make a plan of what libraries you are going to use

state what you understand

state what parts you are fuzzy about
creating a batch of the training image and the label txt file


state what parts you currently have no idea about that you need to learn
how to import the yolo library
how to run the model for classification
how to unfreeze layers for finetuning
how to run the model on gpu

What is the sequence of events that must take place in your program for it to run successfully
run the model on the test set to see its baseline performance
plot the map metric that will measure performance
unfreeze some layers
finetune the model on the training set
use the test set to evaluate the performance of the model
plot the performance of the model (map plot)

What is the evaluation metric that you will use to measure success
map@0.5:0.95
