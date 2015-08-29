package com.kylecorry.lann;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

import com.kylecorry.lann.activation.Activation;
import com.kylecorry.lann.activation.Linear;
import com.kylecorry.lann.activation.Sigmoid;

public class NeuralNetwork {

	private int numLayers;
	private int numNeurons;
	private Layer layers[];
	private boolean verbose = true;

	/**
	 * A representation of a Feed-Forward neural network.
	 * 
	 * @param layers
	 *            An array of integers where the length indicates the number of
	 *            layers and the value indicates the number of neurons at the
	 *            layer with the value's index.
	 * @param functions
	 *            An array of Activation functions where the function at each
	 *            index indicates the activation function that the layer will
	 *            use.
	 */
	public NeuralNetwork(int layers[], Activation functions[]) {
		assert(layers.length == functions.length);
		this.numLayers = layers.length;
		this.numNeurons = Utils.sum(layers);
		this.layers = new Layer[numLayers];
		for (int i = 0; i < numLayers; i++) {
			if (i >= 1) {
				this.layers[i] = new Layer(layers[i], functions[i], getRandomWeights(layers[i], this.layers[i - 1]));
			} else {
				this.layers[i] = new Layer(layers[i], functions[i], getOnes(layers[i]));
			}
		}
	}

	/**
	 * Represents a Feed-Forward neural network.
	 * 
	 * Uses Linear, Sigmoid, Sigmoid as layers.
	 * 
	 * @param numInput
	 *            The number of inputs.
	 * @param numHidden
	 *            The number of hidden neurons.
	 * @param numOutput
	 *            The number of outputs.
	 */
	public NeuralNetwork(int numInput, int numHidden, int numOutput) {
		this(new int[] { numInput, numHidden, numOutput },
				new Activation[] { new Linear(), new Sigmoid(), new Sigmoid() });
	}

	/**
	 * Represents a Feed-Forward neural network.
	 * 
	 * Uses Linear, Sigmoid as the input and hidden layers.
	 * 
	 * @param numInput
	 *            The number of inputs.
	 * @param numHidden
	 *            The number of hidden neurons.
	 * @param numOutput
	 *            The number of outputs.
	 * @param outputFunction
	 *            The activation function of the output layer.
	 */
	public NeuralNetwork(int numInput, int numHidden, int numOutput, Activation outputFunction) {
		this(new int[] { numInput, numHidden, numOutput },
				new Activation[] { new Linear(), new Sigmoid(), outputFunction });
	}

	/**
	 * The number of layers in the neural network including the input and
	 * output.
	 * 
	 * @return The total number of layers in the neural network.
	 */
	public int size() {
		return numLayers;
	}

	/**
	 * The number of neurons in the neural network.
	 * 
	 * @return The total number of neurons in the neural network.
	 */
	public int getNumNeurons() {
		return numNeurons;
	}

	/**
	 * See if the training function prints the result of each iteration.
	 * 
	 * @return The verbosity of the training function.
	 */
	public boolean isVerbose() {
		return verbose;
	}

	/**
	 * Set the verbosity of the training function.
	 * 
	 * @param verbose
	 *            The new verbosity value.
	 */
	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}

	/**
	 * Generates a set of random weights.
	 * 
	 * @param i
	 *            Number of neurons in current layer
	 * @param prevLayer
	 *            The previous layer
	 * @return A set of random weights for the current layer
	 */
	private double[][] getRandomWeights(int i, Layer prevLayer) {
		double weights[][] = new double[i][prevLayer.numNeurons + 1];
		for (int j = 0; j < i; j++) {
			for (int k = 0; k <= prevLayer.numNeurons; k++) {
				weights[j][k] = 2 * Math.random() - 1;
			}
		}
		return weights;
	}

	/**
	 * Generates a weights set consisting of only 1s (for input layer).
	 * 
	 * @param i
	 *            Number of neurons in the input layer.
	 * @return A set of weights equal to 1 for the current layer.
	 */
	private double[][] getOnes(int i) {
		double weights[][] = new double[i][1];
		for (int j = 0; j < i; j++) {
			weights[j] = new double[] { 1 };
		}
		return weights;
	}

	/**
	 * Give a prediction based on some input.
	 * 
	 * @param input
	 *            The input to the neural network which is equal in size to the
	 *            number of input neurons.
	 * @return The output of the neural network or an empty array if input size
	 *         does not equal the length of the first layer.
	 */
	public double[] predict(double input[]) {
		if (layers[0].numNeurons != input.length)
			return new double[] {};
		for (int i = 0; i < layers[0].numNeurons; i++) {
			layers[0].neurons[i].input = input[i];
			layers[0].neurons[i].activate();
		}
		for (int i = 1; i < numLayers; i++) {
			for (int n = 0; n < layers[i].numNeurons; n++) {
				layers[i].neurons[n].input = 0;
				for (int j = 0; j <= layers[i - 1].numNeurons; j++) {
					layers[i].neurons[n].input += layers[i - 1].neurons[j].output * layers[i].neurons[n].weights[j];
				}
				layers[i].neurons[n].activate();
			}
		}

		double output[] = new double[layers[numLayers - 1].numNeurons];
		for (int i = 0; i < output.length; i++) {
			output[i] = layers[numLayers - 1].neurons[i].output;
		}
		return output;
	}

	/**
	 * Train the neural network to predict an output given some input.
	 * 
	 * @param input
	 *            The input to the neural network.
	 * @param output
	 *            The target output for the given input.
	 * @param epochs
	 *            The number of training cycles.
	 * @return The error of the network as an accumulated RMS.
	 */
	public double train(double input[][], double output[][], int epochs) {
		assert(input.length != output.length);
		double error = 0;
		for (int i = 0; i < epochs; i++) {
			error = train(input, output);
			if (verbose)
				System.out.println((i + 1) + " -- " + error);
		}
		return error;
	}

	/**
	 * Train the neural network to predict an output given some input.
	 * 
	 * @param input
	 *            The input to the neural network.
	 * @param output
	 *            The target output for the given input.
	 * @param epochs
	 *            The maximum number of training cycles.
	 * @param acceptableError
	 *            The amount of error at which to stop training.
	 * @return The error of the network as an accumulated RMS.
	 */
	public double train(double input[][], double output[][], int epochs, double acceptableError) {
		assert(input.length != output.length);
		double error = 0;
		for (int i = 0; i < epochs; i++) {
			error = train(input, output);
			if (verbose)
				System.out.println((i + 1) + " -- " + error);
			if (error <= acceptableError)
				break;
		}
		return error;
	}

	/**
	 * Train the neural network to predict an output given some input.
	 * 
	 * @param input
	 *            The input to the neural network.
	 * @param output
	 *            The target output for the given input.
	 * @return The error of the network as an accumulated RMS.
	 */
	public double train(double input[][], double output[][]) {
		assert(input.length != output.length);
		double totalError = 0;
		// calculate overall net error
		for (int i = 0; i < input.length; i++) {
			double error = 0;
			double[] netOut = this.predict(input[i]);
			// double[] outputError = new double[netOut.length];
			for (int j = 0; j < netOut.length; j++) {
				error += Math.pow(output[i][j] - netOut[j], 2);
			}
			error /= layers[numLayers - 1].numNeurons;
			error = Math.sqrt(error);

			for (int n = 0; n < layers[numLayers - 1].numNeurons; n++) {
				layers[numLayers - 1].neurons[n].calcOutputGradients(output[i][n]);
			}

			for (int l = numLayers - 2; l > 0; l--) {

				for (int n = 0; n < layers[l].numNeurons + 1; n++) {
					layers[l].neurons[n].calcHiddenGradients(layers[l + 1], n);
				}

			}

			for (int l = numLayers - 1; l > 0; l--) {
				for (int n = 0; n < layers[l].numNeurons; n++) {
					layers[l].neurons[n].updateInputWeights(layers[l - 1]);
				}
			}
			totalError += error;

		}
		return totalError;
	}

	/**
	 * Get the current weights of the network's connections.
	 * 
	 * @return The current weights of the network connections.
	 */
	public double[][] getWeights() {
		double weights[][] = new double[numNeurons + numLayers][numNeurons];
		int position = 0;
		for (int l = 0; l < numLayers; l++) {
			for (int n = 0; n <= layers[l].numNeurons; n++) {
				weights[position] = layers[l].neurons[n].weights;
				position++;
			}
		}
		return weights;
	}

	/**
	 * Sets the weights of the network's connections.
	 * 
	 * @param weights
	 *            The new weights of the network.
	 */
	public void setWeights(double weights[][]) {
		int position = 0;
		for (int l = 0; l < numLayers; l++) {
			for (int n = 0; n <= layers[l].numNeurons; n++) {
				layers[l].neurons[n].weights = weights[position];
				position++;
			}
		}
	}

	/**
	 * Saves the neural network to a file (CSV).
	 * 
	 * @param filename
	 *            The filename in which to save the weights to.
	 */
	public void save(String filename) {
		PrintWriter printWriter;
		try {
			printWriter = new PrintWriter(filename, "UTF-8");
			for (double[] neuron : getWeights()) {
				for (int i = 0; i < neuron.length; i++) {
					if (i < neuron.length - 1)
						printWriter.print(neuron[i] + ",");
					else
						printWriter.print(neuron[i]);
				}
				printWriter.println();
			}
			printWriter.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Loads a neural network from a file.
	 * 
	 * @param filename
	 *            The name of the file to retrieve the weights from.
	 */
	public void load(String filename) {
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(filename));

			StringBuilder sb = new StringBuilder();
			String line = br.readLine();

			while (line != null) {
				sb.append(line);
				sb.append(System.lineSeparator());
				line = br.readLine();
			}
			br.close();
			String everything = sb.toString();
			String[] neurons = everything.split("\n");
			String[][] weights = new String[neurons.length][numNeurons];
			for (int i = 0; i < weights.length; i++) {
				weights[i] = neurons[i].split(",");
			}
			double[][] dWeights = new double[neurons.length][numNeurons];
			for (int i = 0; i < dWeights.length; i++) {
				for (int n = 0; n < weights[i].length; n++) {
					dWeights[i][n] = Double.valueOf(weights[i][n]);
				}
			}
			setWeights(dWeights);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Rounds the output of a Sigmoid output layer to the nearest whole number.
	 * 
	 * @param input
	 *            The input of the neural network.
	 * @return The output of the neural network with integer values.
	 */
	public double[] classify(double input[]) {
		double[] output = this.predict(input);
		for (int i = 0; i < output.length; i++) {
			output[i] = Math.round(output[i]);
		}
		return output;
	}

	/**
	 * Test the neural network on a testing set.
	 * 
	 * @param input
	 *            The input to the neural network.
	 * @param output
	 *            The target output for the given input.
	 * @return The error of the network as an accumulated RMS.
	 */
	public double test(double input[][], double output[][]) {
		assert(input.length != output.length);
		double totalError = 0.0;
		for (int i = 0; i < input.length; i++) {
			double error = 0.0;
			double newOutput[] = predict(input[i]);
			for (int j = 0; j < output[i].length; j++) {
				error += Math.pow(newOutput[j] - output[i][j], 2);
			}
			error /= output.length;
			error = Math.sqrt(error);
			totalError += error;
		}
		return totalError;
	}

	public static class Builder {
		private ArrayList<Integer> layers;
		private ArrayList<Activation> functions;
		private int inputLayer = 1;
		private Activation inputFunction = new Linear();
		private int outputLayer = 1;
		private Activation outputFunction = new Sigmoid();
		private double[][] weights = null;
		private String weightFile = null;

		public Builder() {
			layers = new ArrayList<Integer>();
			functions = new ArrayList<Activation>();
		}

		public Builder setInputLayer(int numInput) {
			inputLayer = numInput;
			return this;
		}

		public Builder setInputLayer(int numInput, Activation function) {
			inputLayer = numInput;
			inputFunction = function;
			return this;
		}

		public Builder loadWeights(String filename) {
			weightFile = filename;
			return this;
		}

		public Builder setWeights(double[][] weights) {
			this.weights = weights;
			return this;
		}

		public Builder setOutputLayer(int numOutput) {
			outputLayer = numOutput;
			return this;
		}

		public Builder setOutputLayer(int numOutput, Activation function) {
			outputLayer = numOutput;
			outputFunction = function;
			return this;
		}

		public Builder addHiddenLayer(int numNeurons) {
			layers.add(numNeurons);
			functions.add(new Linear());
			return this;
		}

		public Builder addHiddenLayer(int numNeurons, Activation function) {
			layers.add(numNeurons);
			functions.add(function);
			return this;
		}

		public NeuralNetwork build() {
			int[] layersInt = new int[layers.size() + 2];
			Activation[] functionsArr = new Activation[functions.size() + 2];
			for (int i = 0; i < layers.size(); i++) {
				layersInt[i + 1] = layers.get(i);
				functionsArr[i + 1] = functions.get(i);
			}
			layersInt[0] = inputLayer;
			functionsArr[0] = inputFunction;
			layersInt[layersInt.length - 1] = outputLayer;
			functionsArr[functionsArr.length - 1] = outputFunction;
			NeuralNetwork network = new NeuralNetwork(layersInt, functionsArr);
			if (weightFile != null)
				network.load(weightFile);
			if (weights != null)
				network.setWeights(weights);
			return network;
		}

	}
}
