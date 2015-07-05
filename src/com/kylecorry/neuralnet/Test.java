package com.kylecorry.neuralnet;

public class Test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		NeuralNetwork net = new NeuralNetwork(new int[] { 1, 4, 1 },
				new Activation[] { new Linear(), new Sigmoid(), new Softplus() });
		double input[][] = new double[][] { { 0.760 }, { 0.600 }, { 0.475 },
				{ 0.774 }, { 0.761 } };
		double output[][] = new double[][] { { 68. }, { 72. }, { 81. },
				{ 66. }, { 67. } };
		//net.train(input, output, 200, 0.001);
		//net.weightsToFile("temperature.csv");
		net.weightsFromFile("temperature.csv");
		System.out.println(net.activate(new double[]{.761})[0]);
	}

}
