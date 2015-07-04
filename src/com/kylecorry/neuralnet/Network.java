package com.kylecorry.neuralnet;

public class Network {

	public int numLayers;
	public int numNeurons;
	public double weights[][][];
	private Activation functions[];
	private Layer layers[];

	public Network(int layers[], Activation functions[]) {
		this.numLayers = layers.length;
		this.numNeurons = Utils.sum(layers);
		this.functions = functions;
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

}
