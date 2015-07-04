package com.kylecorry.neuralnet;

public class Layer {

	public int numNeurons;
	private Activation function;
	public double weights[][];
	public Neuron neurons[];

	public Layer(int numNeurons, Activation function, double weights[][]) {
		this.numNeurons = numNeurons;
		this.function = function;
		this.weights = weights;
		neurons = new Neuron[numNeurons + 1];
		for (int i = 0; i <= numNeurons; i++) {
			if (i == numNeurons) {
				neurons[i] = new Neuron(new double[] { Math.random() }, 1.0, this.function);
			} else {
				neurons[i] = new Neuron(weights[i], 0, this.function);
			}
		}
	}
	
}
