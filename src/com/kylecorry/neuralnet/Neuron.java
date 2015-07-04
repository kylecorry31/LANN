package com.kylecorry.neuralnet;

public class Neuron {
	public double output;
	public double input;
	public double weights[];
	private Activation function;
	public static double learningRate = 0.01;
	public static double momentum = 0.7;
	public double gradient;
	public double deltaWeights[];

	public Neuron(double weights[], double output, Activation function) {
		this.weights = weights;
		this.output = output;
		this.function = function;
		this.deltaWeights = new double[weights.length];
	}

	public void activate() {
		// summation();
		output = function.activate(input);
	}

	public void summation() {
		input = 0;
		for (int i = 0; i < weights.length; i++) {
			input += weights[i];
			// get input of neuron i of previous layer * weight i
		}
	}

	public void calcOutputGradients(double target) {
		double delta = target - output;
		gradient = delta * function.derivitive(output);

	}

	private double sumDOW(Layer nextLayer, int neuronNum) {
		double sum = 0;
		for (int n = 0; n < nextLayer.numNeurons; n++) {
			sum += nextLayer.neurons[n].weights[neuronNum]
					* nextLayer.neurons[n].gradient;
		}
		return sum;
	}

	public void calcHiddenGradients(Layer nextLayer, int neuronNum) {
		double dow = sumDOW(nextLayer, neuronNum);
		gradient = dow * function.derivitive(output);
	}

	public void updateInputWeights(Layer prevLayer) {
		// TODO Auto-generated method stub
		for (int n = 0; n < prevLayer.numNeurons + 1; n++) {
			double oldDeltaWeight = deltaWeights[n];
			double newDeltaWeight = learningRate * prevLayer.neurons[n].output
					* gradient + momentum * oldDeltaWeight;
			deltaWeights[n] = newDeltaWeight;
			weights[n] += newDeltaWeight;
		}
	}

}
