package com.kylecorry.neuralnet;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.file.Files;

public class Network {

	public int numLayers;
	public int numNeurons;
	private Layer layers[];

	public Network(int layers[], Activation functions[]) {
		this.numLayers = layers.length;
		this.numNeurons = Utils.sum(layers);
		this.layers = new Layer[numLayers];
		for (int i = 0; i < numLayers; i++) {
			if (i >= 1) {
				this.layers[i] = new Layer(layers[i], functions[i],
						getRandomWeights(layers[i], this.layers[i - 1]));
			} else {
				this.layers[i] = new Layer(layers[i], functions[i],
						getOnes(layers[i]));
			}
		}
	}

	private double[][] getRandomWeights(int i, Layer prevLayer) {
		double weights[][] = new double[i][prevLayer.numNeurons + 1];
		for (int j = 0; j < i; j++) {
			for (int k = 0; k <= prevLayer.numNeurons; k++) {
				weights[j][k] = 2 * Math.random() - 1;
			}
		}
		return weights;
	}

	private double[][] getOnes(int i) {
		double weights[][] = new double[i][1];
		for (int j = 0; j < i; j++) {
			weights[j] = new double[] { 1 };
		}
		return weights;
	}

	public double[] activate(double input[]) {
		for (int i = 0; i < layers[0].numNeurons; i++) {
			layers[0].neurons[i].input = input[i];
			layers[0].neurons[i].activate();
		}
		for (int i = 1; i < numLayers; i++) {
			for (int n = 0; n < layers[i].numNeurons; n++) {
				layers[i].neurons[n].input = 0;
				for (int j = 0; j <= layers[i - 1].numNeurons; j++) {
					layers[i].neurons[n].input += layers[i - 1].neurons[j].output
							* layers[i].neurons[n].weights[j];
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

	public double train(double input[][], double output[][]) {

		// calculate overall net error
		for (int i = 0; i < input.length; i++) {
			double error = 0;
			double[] netOut = this.activate(input[i]);
			// double[] outputError = new double[netOut.length];
			for (int j = 0; j < netOut.length; j++) {
				error += Math.pow(output[i][j] - netOut[j], 2);
			}
			error /= layers[numLayers - 1].numNeurons;
			error = Math.sqrt(error);

			for (int n = 0; n < layers[numLayers - 1].numNeurons; n++) {
				layers[numLayers - 1].neurons[n]
						.calcOutputGradients(output[i][n]);
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

		}
		return 0.0;
	}

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

	public void setWeights(double weights[][]) {
		int position = 0;
		for (int l = 0; l < numLayers; l++) {
			for (int n = 0; n <= layers[l].numNeurons; n++) {
				layers[l].neurons[n].weights = weights[position];
				position++;
			}
		}
	}

	public void weightsToFile(String filename) {
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
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void weightsFromFile(String filename) {
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
}
