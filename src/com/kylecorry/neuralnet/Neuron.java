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

	/**
	 * A representation of a neuron.
	 * 
	 * @param weights
	 *            The weights for the connections between this neuron and those
	 *            of the previous layer.
	 * @param output
	 *            The output of the neuron.
	 * @param function
	 *            The activation function of the neuron.
	 */
	public Neuron(double weights[], double output, Activation function) {
		this.weights = weights;
		this.output = output;
		this.function = function;
		this.deltaWeights = new double[weights.length];
	}

	/**
	 * Sets the output of the neuron to the value of its activation function
	 * with the input to the neuron.
	 */
	public void activate() {
		output = function.activate(input);
	}

	/**
	 * Calculates the gradient of the output neuron.
	 * 
	 * @param target
	 *            The target value of the neural network.
	 */
	public void calcOutputGradients(double target) {
		double delta = target - output;
		gradient = delta * function.derivative(output);

	}

	/**
	 * Sums the derivatives of the output weights.
	 * 
	 * @param nextLayer
	 *            The next layer.
	 * @param neuronNum
	 *            The index of this neuron in its layer.
	 */
	private double sumDOW(Layer nextLayer, int neuronNum) {
		double sum = 0;
		for (int n = 0; n < nextLayer.numNeurons; n++) {
			sum += nextLayer.neurons[n].weights[neuronNum]
					* nextLayer.neurons[n].gradient;
		}
		return sum;
	}

	/**
	 * Calculates the gradient of the hidden neuron.
	 * 
	 * @param nextLayer
	 *            The next layer.
	 * @param neuronNum
	 *            The index of this neuron in its layer.
	 */
	public void calcHiddenGradients(Layer nextLayer, int neuronNum) {
		double dow = sumDOW(nextLayer, neuronNum);
		gradient = dow * function.derivative(output);
	}

	/**
	 * Updates the input weights to this neuron.
	 * 
	 * @param prevLayer
	 *            The previous layer.
	 */
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
