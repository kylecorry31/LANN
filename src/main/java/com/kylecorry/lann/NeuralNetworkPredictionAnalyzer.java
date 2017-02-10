package com.kylecorry.lann;

import com.kylecorry.matrix.Matrix;

public class NeuralNetworkPredictionAnalyzer {
	public static int argMax(Matrix input) {
		double max = input.max();
		return input.find(max)[0];
	}
}
