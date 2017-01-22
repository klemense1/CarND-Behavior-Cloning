# CarND-Behavior-Cloning

Wow. That was the most interesting project so far with a couple of challenges.

Why so? Two things come to my mind. 
First, one has to collect it’s own data. So you have to think about what you record and which equipment to use. The data plays a huge role in this project’s success, but I will come to that.
Second, the testing and adjusting loop is quite long. In previous projects with classification, one had a specific metric (accuracy) which had to be improved. Here, driving, seeing the car crash and step by step finding reasons for that takes quite some time.

How did I approach the project?
Udacity provided us with some advices. One was to use some students tips (see [1])

Which ones did I rely on?

- Copy the Nividia Pipeline
- Use generators 
- Use 3 images to overfit and thus quickly test your model architecture

I started right from the beginning implementing Nvidia’s model architecture from the paper „End to End Learning for Self-Driving Cars“. 
The model is first normalized in the graph to make life  easier for the optimizer.

    mdl.add(Lambda(lambda x: x/128. - 1, input_shape=IMAGE_SHAPE, name="input"))

The model then consists of 5 convolutional layers with Max-Pooling, Dropout and a rectified linear unit. The dropout prevents the model from overfitting the data, as it drops half of the weights randomly and thus makes the model develop duplicating patterns to be robust against those fallouts.

    mdl.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same',))
    mdl.add(MaxPooling2D(pool_size=(2, 2)))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

Finally I flatten the graph and use 3 dense layers to get to my regression output.

    mdl.add(Flatten())

    mdl.add(Dense(128, activation='relu'))
    mdl.add(Dense(64, activation='relu'))
    mdl.add(Dense(1, name="output"))

To quickly test the model, I first used model.fit() without a generator and picked three images. This allowed me to overfit my model and get a quick feedback, whether the model is able to predict those three angles. If it isn’t, it won’t be able to predict a whole lap either. I moved model.fit() call into the function train_model() and wrote a test case test_model_and_training_without_generator().
I then continued implementing the generator. How so? For that, I had to write a function my_generator() which I am passing to model.fit_generator(). In my_generator(), I am passing a list of dictionaries containing a picture and an angle. The image is then loaded and passed to the generator in batches using „yield“.

What’s the advantage of preprocessing the list before my_generator and not passing the whole csv? I realized that with using three camera views, it is easier to post process the  list. That said, I am reading the paths of all three lists, adjusting the angle for left and right camera and then storing them in the dictionary as {image, angle}, as the model only uses one image.

- Use an analog joystick.

When I first run the training mode, I used my keyboard. But I quickly realized that I could not manage to steer as smoothly as I wanted and thus would not get to a dataset good enough to train my model. I quickly thought about post-processing the steering angle with some running mean or something, but I chose the other way and borrowed a driving wheel for car games from a friend of mine. I recorded three laps and also recorded going back from the border to the inner lane.

Which mistakes did I initially do?

Well, when I finally trained my model and wanted to see how it behaves in the simulator, the car just did not move. After some debugging, I realized that I had resized the input image to half its size before passing it to the model but forgot to do the same in the simulator. Unfortunately I didn’t throw any errors but just did not work.

I first created a model performing classification. With the angles having all sorts of values (but realistically also still having a limited set of possible angles), I decided to convert those angles to N classes and do classification. That said, I passed 30 different angle classes to my model during training. I still think this approach would work, but during finding out why my model was not moving, I changed my model towards regression. This way, the car is able to steer way more smoothly. That means, instead of having probabilities for N classes, I only had one class but used the probability as the angle itself.

What architectural changes did I have to do from classification to regression?

Regression:

    mdl.add(Dense(128, activation='relu'))
    mdl.add(Dense(64, activation='relu'))
    mdl.add(Dense(1))

Classification:

    mdl.add(Dense(128, activation='relu'))
    mdl.add(Dense(64, activation='relu'))
    mdl.add(Dense(NUM_CLASSES), activation=‚softmax‘)

Also, I had to change the cost function using categorical_crossentropy to mean_squared_error, as categorical_crossentropy relies on classes to compute the difference between. I am using the Adam optimizer, experimented with different learning rates finally sticked to the default one, as decay and other parameters are related to that. Also, I wanted to focus on architecture and data and not tweak the parameters too much. The idea behind that is that a good model should work with multiple hyper parameters.

After changing the cost function, I started training the model again and wondered why the accuracy didn’t really improve or at least change. I thought that the generator pipleline might be wrong, but eventually  realized that accuracy is not the right metric with regression any more and that I could only rely on the costs being computed.

Now I could see some steering but always the same output (it slightly varied but only after the 5 decimal or so). Why so? I guess I wasn’t using enough input data at that point to cover. Also, I had forgot to implement the pooling. Although Kaspar Sakman (another Udacity student fellow explaining his approach, see [2]) writes that he does not use pooling for regression, I didn’t experience the same. I implemented the pooling and finally got some steering.

Still, my model did not past the first corner. It had a drift to the left and did not steer enough. The reason is obvious when investigating the input data. I had a lot of slightly left steering but firstly the steering angles were not distributed equally and secondly the high angles were underrepresented.

I came to the conclusion that my input data was not enough. I needed more data. As described by Vivek Yadav in [3], augmentation was the easiest way to get to more data.

I focused on:

- Flipping the data. Thus way, I would get to as much left steering data as right steering data.
- Adjusting the brightness. The first track does not have brightness changes, but the second has.
- Duplicating the data with steering angles larger that 0.2. This way, I would get to more steering data with angles necessary to react in situations when you are close to the side line.
- Slightly changing the steering angle. Modifying, Shearing or rotating the picture would be quite some effort, but why not change the angle and thus generate a different image-angle pair this way?

Still, the model could not get through the whole lap. Did I miss something? I recorded the training data again but did not improve my model much. Finally, I changed to the Udacity training data and … it worked!

## What did I learn?

Data: Shit in, shit out. Although I knew that already, I wasn’t aware of the amount of data necessary to train such a conv neural network. I was even more surprised how accurate one has to be to record the data. Even now, my own recorded training data was not good enough.

## Next steps?

The model does not get through the second route in the simulator. I assume that brightness, shadows and others light effects are challenging my model. The next step is to incorporate that into my augmentation and thus generate more training data.

I still want to test whether classification works as well, although the steering will be less smoothly.

I want to improve the longitudinal driver. I started out with a constant throttle and then changed it to one where the throttle is adaptive to the steering angle. Although the simulator does not have a proper vehicle dynamics model, I would like to make the throttle  depend on steering and previous velocity, as it thus becomes closer to the way the human drove the car when recording the data.

Record my own sufficient training data.

[1] https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet

[2] https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.xqcvfgxxg

[3] https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.2kz9bqk51
