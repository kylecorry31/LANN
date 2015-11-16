package com.kylecorry.lann;

import java.util.ArrayList;

import com.kylecorry.lann.activation.Activation;
import com.kylecorry.matrix.Matrix;

public class NewNeuralNetwork {

	private ArrayList<Layer> layers;

	public NewNeuralNetwork() {
		layers = new ArrayList<Layer>();
	}
	
	void addLayer(Layer l){
		layers.add(l);
	}

	public class Builder {
		private NewNeuralNetwork net;
		public Builder() {
			net = new NewNeuralNetwork();
		}
		
		public NewNeuralNetwork addLayer(int[] size, Activation function){
			Layer l = new Layer(size, function);
			net.addLayer(l);
			return net;
		}
		
		public NewNeuralNetwork build(){
			return net;
		}

	}

	class Layer {
		private double[][] weights;
		private Activation function;
		private double[][] bias;

		/**
		 * 
		 * @param size
		 *            [input, output]
		 * @param fn
		 */
		public Layer(int[] size, Activation fn) {
			// fill weights - # of connections (input * output) weights.length =
			// input
			// weights[0].length = output
			// fill bias - bias.length = output, bias[0].lenght = 1
			weights = new double[size[0]][size[1]];
			bias = new double[size[1]][1];
			function = fn;
		}

		public double[][] activate(double[][] input) {
			return applyFunction(Matrix.matAdd(Matrix.matMult(input, weights), bias));
		}

		private double[][] applyFunction(double[][] input) {
			double[][] activated = input.clone();
			for (int i = 0; i < input.length; i++) {
				for (int j = 0; j < input[i].length; j++) {
					activated[i][j] = function.activate(input[i][j]);
				}
			}
			return activated;
		}
	}

}
