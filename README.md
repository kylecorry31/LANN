# Artificial-Neural-Network

A lightweight implementation of an artificial neural network (feed-forward multi-layer perceptron neural network to be exact).

This is a very simple library to use in your projects! It has very user friendly functions and allows for a neural network to be created very quickly and efficiently.

Getting Started
===============

Only three simple steps to get started!

1. Download [this .jar file](ArtificialNeuralNetwork-Kyle.jar?raw=true) and add it to your build path of your Java project. (Supports JDK 1.6+)

2. Create a NeuralNetwork object and train it as seen in the example below.

3. Use your neural network!

Example
=======

```java
public class Test {

	public static void main(String[] args) {
	
	  	// A neural network to tell me if I should go hiking based on the weather.

		// Create a neural network with 7 input neurons, 10 hidden neurons and 1 output neurons (Sigmoid)
		// In most cases the first layer will use the linear activation function.
		NeuralNetwork net = new NeuralNetwork(new int[] { 7, 10, 1 },
				new Activation[] { new Linear(), new Sigmoid(), new Sigmoid() });

		// The input to the neural network
		/* Each item is an input -- current temperature, current humidity, current pressure, 
		 *	high temperature, low temperature, average pressure, average humidity
		 */
		double input[][] = new double[][] {
				{ 67.08, 59 / 100., 1019, 75.94, 60.24, 1013.94 / 1013.25,
						62 / 100. },
				{ 69.84, 46 / 100., 1023.26 / 1013.25, 79.7, 62.11, 1012.57,
						51 / 100. },
				{ 68.43, 83 / 100., 1014.31, 79.43, 61.03, 1005.16 / 1013.25,
						73 / 100. },
				{ 72.57, 50 / 100., 1025.41 / 1013.25, 81.61, 62.46, 1014.79,
						65 / 100. },
				{ 69.25, 99 / 100., 1009.1 / 1013.25, 76.44, 69.26, 999.05,
						99 / 100. },
				{ 76.89, 65 / 100., 1013.45, 75.94, 70.63, 1002.19 / 1013.25,
						72 / 100. },
				{ 76.44, 44 / 100., 1017, 78.35, 62.35, 1009.5 / 1013.25,
						56 / 100. },
				{ 64.71, 59 / 100., 1022, 71.37, 60.03, 1012.98 / 1013.25,
						60 / 100. },
				{ 72.41, 56 / 100., 1013, 82.17, 71.76, 1006.71 / 1013.25,
						60 / 100. },
				{ 80.94, 47 / 100., 1022, 84.99, 73.51, 1016.15 / 1013.25,
						66 / 100. } };

		// Target output of the neural network of the input above
		// 1 is hike, 0 is don't hike
		double output[][] = new double[][] { { 1 }, { 1 }, { 0 }, { 0 }, { 0 },
				{ 0 }, { 1 }, { 0 }, { 1 }, { 0 } };
				
				
		// Train the network and print the error
		System.out.println(net.train(input, output, 200, 0.01));
		
		// Save the weights to a file
		net.weightsToFile("hike.csv");
		
		// See how well the network is performing by testing on a new set of data.
		System.out.println(net.test(new double[][] {
				{ 82.81, 65 / 100., 1015, 81.14, 62.08, 985.59 / 1013.25,
						46 / 100. },
				{ 86.63, 51 / 100., 1016.7 / 1013.25, 89.74, 85.32, 1022.23,
						82 / 100. } }, new double[][] { { 0.0 }, { 1 } }));
	}
}

```


