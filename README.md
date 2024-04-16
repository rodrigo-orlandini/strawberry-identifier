# Strawberry Identifier 🍓🧠

On the ending of 2022, I was finishing my bachelor's degree and delivering my final paper to be approved.

The project developed in the final paper is an embedded system based on artificial intelligence to identify and catch ripe strawberries in a plantation. You can find more details [here](https://github.com/rodrigo-orlandini/automation-eng-final-paper).

Due to some personal reasons, I had less time than I wanted and I chose to use an AI API from [Roboflow](https://roboflow.com/) instead of studying and creating one model from scratch.

Now, after 2 years, with time to study and a bit of experience in Artificial Intelligence, I decided to create this model to identify ripe strawberries.

### Techonolgies

[![Python](.markdown/python.png "Python")](https://www.python.org/)
[![Tensorflow](.markdown/tensorflow.png "Tensorflow")](https://www.tensorflow.org/)

### Development and Results

To develop this model I used the Tensorflow Object Detection API and tried to train several different model architectures, such as MobileNet, EfficientDet and some ResNet variations. I achieved the best results with CenterNet ResNet and MobileNet, where the accuracy after training reached round of 65%.

The best results I obtained can be visualized in the images bellow, including successful and failed detections:

![Success 01](.markdown/samples/success-01.png)
![Success 02](.markdown/samples/success-02.png)
![Success 03](.markdown/samples/success-03.png)
![Success 04](.markdown/samples/success-04.png)

Here is a failed example, where it detected an apple as a strawberry:

![Failed](.markdown/samples/failed.png)

To avoid this behavior, I included some similar objects and trained the model to categorize them as "Any". I did this for apples, red balls, peaches and red mushrooms.

Keep in mind that I used a common machine (no GPUs and only 8GB RAM) to train the neural network, so I wasn't able to run specific model architectures, and I had to use just a few images (almost 20) for training. 

I couldn't upload trained models to GitHub because of their size. However, you can obtain them using Tensorflow Object Detection API with the pipeline.config files inside models folder, tfrecords in records folder and the label map. Additionally, it's recommended to create a virtual environment to install dependency packages.

### References

[Tensorflow Object Detection Course on Youtube](https://www.youtube.com/watch?v=yqkISICHH-U&t=9438s)

[Tensorflow Object Detection Docs](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md)

[Tensorflow Docs](https://www.tensorflow.org/api_docs)
