package com.kylecorry.lann;

import com.kylecorry.matrix.Matrix;

public abstract class AbstractMachineLearningAlgorithm implements MachineLearningAlgorithm {

	public double fit(Matrix[] input, Matrix[] output, double maxIterations) {
		return fit(input, output, maxIterations, 0.001);
	}

	public double fit(Matrix[] input, Matrix[] output, double maxIterations, double tolerance) {
		double error = Double.POSITIVE_INFINITY;
		for (int i = 0; i < maxIterations; i++) {
			error = train(input, output);
			if (error <= tolerance)
				return error;
		}
		return error;
	}

	public double fit(Matrix[] input, Matrix[] output) {
		return fit(input, output, 1000, 0.001);
	}

	public double accuracy(Matrix[] input, Matrix[] output) {
		return accuracy(input, output, false);
	}

	public double accuracy(Matrix[] input, Matrix[] output, boolean argmax) {
		double score = 0;
		for (int i = 0; i < input.length; i++) {
			Matrix prediction = predict(input[i]);
			if (argmax)
				prediction = prediction.oneHot();
			if (prediction.equals(output[i].transpose()))
				score++;
		}
		if (input.length == 0)
			return 0;
		return score / input.length;
	}

}
