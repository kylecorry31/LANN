package com.kylecorry.lann;

import com.kylecorry.lann.activation.Activation;
import com.kylecorry.lann.activation.Softmax;
import com.kylecorry.matrix.Matrix;

import java.io.*;
import java.security.InvalidParameterException;
import java.util.ArrayList;

@Deprecated
public class NeuralNetwork {

	private ArrayList<Layer> layers;
	private double lambda = 0;

	/**
	 * A representation of a Feed-Forward neural network.
	 */
	NeuralNetwork() {
		layers = new ArrayList<Layer>();
	}

	/**
	 * Add a layer to the neural network.
	 * 
	 * @param l
	 *            The layer to add to the neural network.
	 */
	private void addLayer(Layer l) {
		if (layers.size() == 0
				|| layers.get(layers.size() - 1).getLayerSize().getOutputSize() == l.getLayerSize().getInputSize()) {
			layers.add(l);
		} else {
			System.err.println("Layer input did not match the output of the last layer.");
			System.exit(1);
		}
	}

	/**
	 * Calculate the cross entropy error of the neural network.
	 * 
	 * @param x
	 *            The input to the neural network.
	 * @param y
	 *            The expected output.
	 * @return The cross entropy error.
	 */
	public double crossEntropyError(Matrix x, Matrix y) {
		Matrix y_ = predict(x);
		double j = y_.multiply(y.map(new Matrix.Function() {
			@Override
			public double function(double x) {
				return Math.log(x);
			}
		})).sum();
		return -j;
	}

	/**
	 * Calculate the squared error of the neural network.
	 * 
	 * @param x
	 *            The input to the neural network.
	 * @param y
	 *            The expected output.
	 * @return The squared error.
	 */
	public double squaredError(Matrix x, Matrix y) {
		Matrix y_ = predict(x);
		double sumSquareWeights = 0;
		for (Layer layer : layers)
			sumSquareWeights += layer.weightMatrix.power(2).sum();
		double j = 0.5 * y_.subtract(y).power(2).sum() / layers.get(0).getLayerSize().getInputSize()
				+ lambda / 2 * sumSquareWeights;
		return j;
	}

	/**
	 * Give a prediction based on some input.
	 * 
	 * @param input
	 *            The input to the neural network which is equal in size to the
	 *            number of input neurons.
	 * @return The output of the neural network.
	 */
	public Matrix predict(Matrix input) {
		if (input.getNumRows() != layers.get(0).getLayerSize().getInputSize()) {
			throw new InvalidParameterException("Input size did not match the input size of the first layer");
		}
		Matrix modInput = (Matrix) input.clone();
		for (Layer l : layers) {
			modInput = l.activate(modInput);
		}
		return modInput;
	}

	/**
	 * Get the position of the most probable in an output array.
	 * 
	 * @param output
	 *            The output of the neural network (using Softmax)
	 * @return The position of the most probable class.
	 */
	public static int argMax(Matrix output) {
		double max = output.max();
		return output.find(max)[0];
	}

	/**
	 * Get the weight matrix of a layer of the neural network.
	 * 
	 * @param layer
	 *            The layer number of the neural network.
	 * @return The weights of the corresponding layer.
	 */
	public Matrix getWeights(int layer) {
		return (Matrix) layers.get(layer).weightMatrix.clone();
	}

	/**
	 * Set the weight matrix of a layer of the neural network.
	 * 
	 * @param layer
	 *            The layer number of the neural network.
	 * @param weights
	 *            The new weight matrix for the layer.
	 */
	public void setWeights(int layer, Matrix weights) {
		layers.get(layer).weightMatrix = (Matrix) weights.clone();
	}

	/**
	 * Get the size of the neural network.
	 * 
	 * @return The number of layers in the neural network.
	 */
	public int size() {
		return layers.size();
	}

	/**
	 * Train the neural network to predict an output given some input.
	 * 
	 * @param input
	 *            The input to the neural network.
	 * @param output
	 *            The target output for the given input.
	 * @param learningRate
	 *            The rate at which the neural network learns. This is normally
	 *            0.01.
	 * @return The error of the network as an mean cross entropy.
	 */
	public double train(Matrix input, Matrix output, double learningRate) {
		double totalError = 0;
		if (input.getNumRows() == output.getNumRows()) {
			for (int i = 0; i < input.getNumRows(); i++) {
				Matrix inputRow = new Matrix(new double[][] { input.getRow(i) }).transpose();
				Matrix outputRow = new Matrix(new double[][] { output.getRow(i) }).transpose();
				Matrix netOutput = this.predict(inputRow);
				// Output layer
				Matrix previousDelta = outputRow.subtract(netOutput).multiply(-1).multiply(layers.get(layers.size() - 1)
						.applyFunctionDerivative(layers.get(layers.size() - 1).inputMatrix));
				Matrix change = previousDelta.dot(layers.get(layers.size() - 2).outputMatrix.transpose())
						.add(layers.get(layers.size() - 1).weightMatrix.multiply(lambda));
				layers.get(layers.size() - 1).weightMatrix = layers.get(layers.size() - 1).weightMatrix
						.subtract(change.multiply(learningRate));
				// Hidden layers
				for (int l = layers.size() - 2; l > 0; l--) {
					previousDelta = layers.get(l + 1).weightMatrix.transpose().dot(previousDelta)
							.multiply(layers.get(l).applyFunctionDerivative(layers.get(l).inputMatrix));
					change = previousDelta.dot(layers.get(l - 1).outputMatrix.transpose())
							.add(layers.get(l).weightMatrix.multiply(lambda));
					layers.get(l).weightMatrix = layers.get(l).weightMatrix.subtract(change.multiply(learningRate));
				}
				double error = squaredError(inputRow, outputRow);
				totalError += error;
			}
		}
		return totalError;
	}

	/**
	 * Saves the neural network to a file (CSV).
	 * 
	 * @param file
	 *            The file in which to save the weights to.
	 */
	public void save(File file) {
		PrintWriter printWriter;
		try {
			printWriter = new PrintWriter(file, "UTF-8");
			for (Layer l : layers) {
				printWriter.println(l.weightMatrix.toString().replace("\n", ""));
			}
			printWriter.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
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
			for (Layer l : layers) {
				printWriter.println(l.weightMatrix.toString().replace("\n", ""));
			}
			printWriter.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Loads a neural network from an InputStream.
	 * 
	 * @param is
	 *            The InputStream to retrieve the weights from.
	 */
	public void load(InputStream is) {
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(is, "utf-8"));
			loadFromBufferedReader(br);
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}

	}

	/**
	 * Loads a neural network from a BufferedReader.
	 * 
	 * @param br
	 *            The buffered reader to retrieve the weights from.
	 */
	private void loadFromBufferedReader(BufferedReader br) {
		try {
			StringBuilder sb = new StringBuilder();
			String line = br.readLine();

			while (line != null) {
				sb.append(line);
				sb.append(System.lineSeparator());
				line = br.readLine();
			}
			br.close();
			String everything = sb.toString();
			String[] strLayers = everything.split("\n");
			for (int i = 0; i < strLayers.length; i++) {
				String[] rows = strLayers[i].split("\\]\\[");
				for (int r = 0; r < rows.length; r++) {
					String[] cols = rows[r].replace("[", "").replace("]", "").split(", ");
					for (int c = 0; c < cols.length; c++) {
						layers.get(i).weightMatrix.set(r, c, Double.parseDouble(cols[c]));
					}
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Loads a neural network from a file.
	 * 
	 * @param file
	 *            The file to retrieve the weights from.
	 */
	public void load(File file) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			loadFromBufferedReader(br);
		} catch (FileNotFoundException e) {
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
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			loadFromBufferedReader(br);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}

	/**
	 * Builder for creating neural network instances.
	 */
	public static class Builder {
		private NeuralNetwork net;

		public Builder() {
			net = new NeuralNetwork();
		}

		/**
		 * Adds a layer to the neural network.
		 * 
		 * @param size
		 *            The size of the layer.
		 * @param function
		 *            The activation function of the layer.
		 */
		public NeuralNetwork.Builder addLayer(int input, int output, Activation function) {
			Layer l = new Layer(new LayerSize(input, output), function);
			net.addLayer(l);
			return this;
		}

		public NeuralNetwork.Builder setRegularization(double lambda) {
			net.lambda = lambda;
			return this;
		}

		/**
		 * Builds a neural network instance.
		 */
		public NeuralNetwork build() {
			return net;
		}

	}

	static class LayerSize {
		private int input, output;

		/**
		 * Represents the size of a layer, with input and output sizes.
		 * 
		 * @param input
		 *            The size of the input.
		 * @param output
		 *            The size of the output.
		 */
		public LayerSize(int input, int output) {
			this.input = input;
			this.output = output;
		}

		/**
		 * The input size of the layer.
		 * 
		 * @return The size of the input.
		 */
		public int getInputSize() {
			return input;
		}

		/**
		 * The output size of the layer.
		 * 
		 * @return The size of the output.
		 */
		public int getOutputSize() {
			return output;
		}
	}

	static class Layer {
		private Matrix weightMatrix, biasMatrix, outputMatrix, inputMatrix;
		private Activation function;
		LayerSize layerSize;

		/**
		 * Represents a layer in a neural network.
		 * 
		 * @param size
		 *            The size of the layer.
		 * @param function
		 *            The activation function for the neurons in this layer.
		 */
		public Layer(LayerSize size, Activation fn) {
			weightMatrix = new Matrix(size.getOutputSize(), size.getInputSize());
			biasMatrix = new Matrix(size.getOutputSize(), 1, 0.1);
			outputMatrix = new Matrix(size.getOutputSize(), 1);
			inputMatrix = new Matrix(size.getInputSize(), 1);
			function = fn;
			layerSize = size;
			weightMatrix = createRandomMatrix(size.getOutputSize(), size.getInputSize());
		}

		private Matrix createRandomMatrix(int rows, int cols) {
			Matrix random = new Matrix(rows, cols);
			return random.map(new Matrix.Function() {

				@Override
				public double function(double x) {
					return Math.random();
				}
			});
		}

		private Matrix applyFunctionDerivative(Matrix input) {
			Matrix activated = (Matrix) input.clone();
			if (function instanceof Softmax)
				activated = activated.map(new Matrix.Function() {

					@Override
					public double function(double x) {
						return Math.exp(x);
					}
				});
			else
				activated = activated.map(new Matrix.Function() {

					@Override
					public double function(double x) {
						return function.derivative(x);
					}
				});

			if (function instanceof Softmax) {
				double sum = activated.sum();
				if (sum != 0)
					activated = activated.multiply(1 / sum);
				activated = activated.subtract(input);
				activated = activated.map(new Matrix.Function() {

					@Override
					public double function(double x) {
						return function.activate(x);
					}
				});
			}
			return activated;
		}

		/**
		 * Processes the input to the layer.
		 * 
		 * @param input
		 *            The input to the layer.
		 * @return The output of the layer.
		 */
		private Matrix activate(Matrix input) {
			inputMatrix = weightMatrix.dot(input).add(biasMatrix);
			Matrix y = applyFunction(inputMatrix);
			outputMatrix = y;
			return y;
		}

		/**
		 * Applies the activation function to the processed input.
		 * 
		 * @param input
		 *            The input to the activation function.
		 * @return The output of the activation function.
		 */
		private Matrix applyFunction(Matrix input) {
			Matrix activated = (Matrix) input.clone();
			for (int row = 0; row < input.getNumRows(); row++)
				for (int col = 0; col < input.getNumCols(); col++)
					activated.set(row, col, function.activate(input.get(row, col)));
			if (function instanceof Softmax) {
				double sum = activated.sum();
				if (sum != 0)
					activated = activated.multiply(1 / sum);
			}
			return activated;
		}

		/**
		 * Get the input and output size of the layer.
		 * 
		 * @return A LayerSize object representing the size of the layer.
		 */
		public LayerSize getLayerSize() {
			return layerSize;
		}

	}

}
