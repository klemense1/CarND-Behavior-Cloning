# CarND-Behavior-Cloning

used tips from here
https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet

Model Architecture

[Has an appropriate model architecture been employed for the task?
The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.]

- type of model used
- the number of layers
- the size of each layer.
- visualizations

Early Testing of Architecture
- tried to overfit the model using three pictures (of course, they have to be representative)
- used fit instead of fit_generator here, as three pictures are easy to load

Regression vs Classification
- first tried classification
- number of classes : 20
- softmax
- should work as well, but steering is not smooth then with selected number of classes

Necessary changes:
Kostenfunktion verwendete categorical_crossentropy —> geht bei Regression nicht

Regression:
    mdl.add(Dense(128, activation='relu'))
    mdl.add(Dense(64, activation='relu'))
    mdl.add(Dense(1))

Classification:
    mdl.add(Dense(128, activation='relu'))
    mdl.add(Dense(64, activation='relu'))
    mdl.add(Dense(NUM_CLASSES), activation=‚softmax‘)

[Has an attempt been made to reduce overfitting of the model?
Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.]



[Have the model parameters been tuned appropriately?
Learning rate parameters are chosen with explanation, or an Adam optimizer is used.]
Adam optimizer is used providing sufficient results


[Is the training data chosen appropriately?
Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).]

- characteristics of the dataset 
- how was dataset generated
	- first used a keyboard
	- than moved on to a driving wheel
- examples of images from the dataset