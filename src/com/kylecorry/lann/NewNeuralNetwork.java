package com.kylecorry.lann;

import java.util.ArrayList;

import com.kylecorry.lann.activation.Activation;
import com.kylecorry.matrix.Matrix;

class NewNeuralNetwork {

	private ArrayList<Layer> layers;

	/**
	 * A representation of a Feed-Forward neural network.
	 */
	NewNeuralNetwork() {
		layers = new ArrayList<Layer>();
	}

	/**
	 * Add a layer to the neural network.
	 * 
	 * @param l
	 *            The layer to add to the neural network.
	 */
	private void addLayer(Layer l) {
		if (layers.size() == 0 || layers.get(layers.size() - 1).getSize()[1] == l.getSize()[0]) {
			layers.add(l);
		} else {
			System.err.println("Layer input did not match the output of the last layer.");
			System.exit(1);
		}
	}

	/**
	 * Calculate the cross entropy error of the neural network.
	 * 
	 * @param y
	 *            The output of the neural network.
	 * @param y_
	 *            The actual expected output.
	 * @return The cross entropy error.
	 */
	public double crossEntropyError(double[] y, double[] y_) {
		double sum = 0;
		for (int i = 0; i < y.length; i++) {
			sum += y_[i] * Math.log(y[i]);
		}
		return -sum;
	}

	/**
	 * Give a prediction based on some input.
	 * 
	 * @param input
	 *            The input to the neural network which is equal in size to the
	 *            number of input neurons.
	 * @return The output of the neural network.
	 */
	public double[] predict(double[] input) {
		if (input.length != layers.get(0).getSize()[0]) {
			System.err.println("Input size did not match the input size of the first layer.");
			System.exit(1);
		}
		double[][] modInput = Matrix.transpose(new double[][] { input });
		for (Layer l : layers) {
			modInput = l.activate(modInput);
		}
		return Matrix.transpose(modInput)[0];
	}

	/**
	 * Give a prediction based on some input with probability percent.
	 * 
	 * @param input
	 *            The input to the neural network which is equal in size to the
	 *            number of input neurons.
	 * @return The output of the neural network.
	 */
	public double[] classify(double[] input) {
		return normalize(predict(input));
	}

	private double[] normalize(double[] input) {
		double min = 0, max = 0;
		for (double i : input) {
			min = Math.min(i, min);
			max = Math.max(i, max);
		}
		double[] modInput = input.clone();
		for (int i = 0; i < input.length; i++) {
			modInput[i] = (input[i] - min) / (max - min);
		}
		return modInput;
	}

	/**
	 * Builder for creating neural network instances.
	 */
	public static class Builder {
		private NewNeuralNetwork net;

		public Builder() {
			net = new NewNeuralNetwork();
		}

		/**
		 * Adds a layer to the neural network.
		 * 
		 * @param size
		 *            The size of the layer in this format: [input, output]
		 * @param function
		 *            The activation function of the layer.
		 */
		public NewNeuralNetwork.Builder addLayer(int[] size, Activation function) {
			Layer l = new Layer(size, function);
			net.addLayer(l);
			return this;
		}

		/**
		 * Builds a neural network instance.
		 */
		public NewNeuralNetwork build() {
			return net;
		}

	}

	static class Layer {
		private double[][] weights;
		private Activation function;
		private double[][] bias;
		private int[] size;

		/**
		 * Represents a layer in a neural network.
		 * 
		 * @param size
		 *            The size of the layer in this format: [input, output]
		 * @param function
		 *            The activation function for the neurons in this layer.
		 */
		public Layer(int[] size, Activation fn) {
			weights = new double[size[1]][size[0]];
			bias = new double[size[1]][1];
			function = fn;
			this.size = size;
			initializeWeights();
			initializeBias();
		}

		/**
		 * Initializes the layer's weights to random double values between 0 and
		 * 1.
		 */
		private void initializeWeights() {
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] = Math.random();
				}
			}
		}

		/**
		 * Initializes the layer's bias neurons to random double values between
		 * 0 and 1.
		 */
		private void initializeBias() {
			for (int i = 0; i < bias.length; i++) {
				for (int j = 0; j < bias[i].length; j++) {
					bias[i][j] = Math.random();
				}
			}
		}

		/**
		 * Processes the input to the layer.
		 * 
		 * @param input
		 *            The input to the layer.
		 * @return The output of the layer.
		 */
		public double[][] activate(double[][] input) {
			return applyFunction(Matrix.matAdd(Matrix.matMult(weights, input), bias));
		}

		/**
		 * Applies the activation function to the processed input.
		 * 
		 * @param input
		 *            The input to the activation function.
		 * @return The output of the activation function.
		 */
		private double[][] applyFunction(double[][] input) {
			double[][] activated = input.clone();
			for (int i = 0; i < input.length; i++) {
				for (int j = 0; j < input[i].length; j++) {
					activated[i][j] = function.activate(input[i][j]);
				}
			}
			return activated;
		}

		/**
		 * Get the input and output size of the layer.
		 * 
		 * @return An int array in this format: [input, output] representing the
		 *         size of the layer.
		 */
		public int[] getSize() {
			return size;
		}

	}

}
