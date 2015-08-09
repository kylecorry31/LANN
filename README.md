# LANN

> Lightweight artificial neural network implementation for Java.

![LANN](http://kylecorry31.github.io/res/images/LANN.jpg)

## Features

* Feedforward prediction
* Backprop training
* Scalable design - create as many neurons / layers as needed

## Getting Started

### Download

Download the .jar file from the [releases page](https://github.com/kylecorry31/Kynet/releases) and add it to your build path of your Java project. (Supports JDK 1.6+)

### Example - Hiking Based on Weather

```java
		// Create
		NeuralNetwork net = new NeuralNetwork(new int[] { 7, 10, 1 },
				new Activation[] { new Linear(), new Sigmoid(), new Sigmoid() });

		// Training input
		double input[][] = new double[][] {
				{ 67.08, 59 / 100., 1019 / 1013.25, 75.94, 60.24, 1013.94 / 1013.25,
						62 / 100. },
				{ 69.84, 46 / 100., 1023.26 / 1013.25, 79.7, 62.11, 1012.57 / 1013.25,
						51 / 100. },
				{ 68.43, 83 / 100., 1014.31 / 1013.25, 79.43, 61.03, 1005.16 / 1013.25,
						73 / 100. },
				{ 72.57, 50 / 100., 1025.41 / 1013.25, 81.61, 62.46, 1014.79 / 1013.25,
						65 / 100. },
				{ 69.25, 99 / 100., 1009.1 / 1013.25, 76.44, 69.26, 999.05 / 1013.25,
						99 / 100. },
				{ 76.89, 65 / 100., 1013.45 / 1013.25, 75.94, 70.63, 1002.19 / 1013.25,
						72 / 100. },
				{ 76.44, 44 / 100., 1017 / 1013.25, 78.35, 62.35, 1009.5 / 1013.25,
						56 / 100. },
				{ 64.71, 59 / 100., 1022 / 1013.25, 71.37, 60.03, 1012.98 / 1013.25,
						60 / 100. },
				{ 72.41, 56 / 100., 1013 / 1013.25, 82.17, 71.76, 1006.71 / 1013.25,
						60 / 100. },
				{ 80.94, 47 / 100., 1022 / 1013.25, 84.99, 73.51, 1016.15 / 1013.25,
						66 / 100. } };

		// Target output
		double output[][] = new double[][] { { 1 }, { 1 }, { 0 }, { 0 }, { 0 },
				{ 0 }, { 1 }, { 0 }, { 1 }, { 0 } };
				
		// Train
		System.out.println(net.train(input, output, 200, 0.01));
		
		// Save
		net.weightsToFile("hike.csv");
		
		// Predict
		System.out.println(net.predict(
		    new double[] { 82.81, 65 / 100., 1015 / 1013.25, 81.14, 62.08, 985.59 / 1013.25, 46 / 100.}
		));

```
