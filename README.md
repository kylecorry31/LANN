# LANN

[![Build Status](https://travis-ci.org/kylecorry31/LANN.svg?branch=master)](https://travis-ci.org/kylecorry31/LANN)

LANN is a lightweight artificial neural network implementation for Java. This library can be used to add simple machine learning models to your program. The API is designed to be simple and easy to use to allow you to immediately get started. LANN provides interfaces which will allow you to swap around the implementation algorithm quickly, and it also includes several trainers such as genetic and backpropagation algorithms. If you just need a classifier or regressor model, those are built in by default (Classifier or NN - you'll need to have a Linear activation function as the output layer for now).

This library is under active development, and some classes may change.


## Installation
To install this library into your program, just include this jar from the releases page in your build path.

## Usage
### Create a neural network
```java
// Create a neural network with a Sigmoid input layer and Softmax output layer (input size = 2, hidden size = 4, output size = 3)
PersistentMachineLearningAlgorithm testNet = new NN.Builder()
        .addLayer(2, 4, new Sigmoid())
        .addLayer(4, 3, new Softmax())
        .build();
```

### Train a neural network
```java
// testNet: input size = 2, output size = 3

Matrix[] inputData = new Matrix[]{new Matrix(100d, 2d), new Matrix(0d, 10d)};
Matrix[] outputData = new Matrix[]{new Matrix(1d, 0d, 0d), new Matrix(0d, 1d, 0d)};

testNet.fit(inputData, outputData);
```

### Predict with a neural network
```java
// testNet: input size = 2, output size = 3
Matrix prediction = testNet.predict(100d, 2d);
```

### Use a neural network as a classifier
```java
// testNet: input size = 2, output size = 3, output layer = Softmax

// Create the classifier from a network
IClassifier<String> classifier = new Classifier<>(testNet, new String[]{"One", "Two", "Three"});

// Predict the class of the input
String classification = classifier.classify(new Matrix(100d, 2d)).getClassification();
```


## Contributing
Please fork this repo and submit a pull request to contribute. I will review all changes and respond if they are accepted or rejected (as well as reasons, so it will be accepted).

## Credits
Just me for now, help is always great!

## License
This project is published under the GPL-3.0 license. Please refer to [LICENSE](LICENSE) for more details.
