package com.kylecorry.lann;

import com.kylecorry.matrix.Matrix;

public interface MachineLearningAlgorithm {

    /**
     * Make a prediction given an input.
     *
     * @param input The input of the machine learning algorithm.
     * @return The prediction.
     */
    Matrix predict(Matrix input);

    /**
     * Make a prediction given an input.
     *
     * @param input The input of the machine learning algorithm.
     * @return The prediction.
     */
    Matrix predict(double... input);

    /**
     * Train the machine learning algorithm to better predict an output given an input. This performs a single iteration.
     *
     * @param input  The input of the algorithm.
     * @param output The desired output of the algorithm.
     * @return The error of the training iteration.
     */
    double train(Matrix[] input, Matrix[] output);

    /**
     * Fit the machine learning algorithm to an input and output data set.
     *
     * @param input         The input of the algorithm.
     * @param output        The desired output of the algorithm in the same order as the input.
     * @param maxIterations The max number of training iterations to perform.
     * @return The error of the last training iteration.
     */
    double fit(Matrix[] input, Matrix[] output, double maxIterations);

    /**
     * Fit the machine learning algorithm to an input and output data set.
     *
     * @param input         The input of the algorithm.
     * @param output        The desired output of the algorithm in the same order as the input.
     * @param maxIterations The max number of training iterations to perform.
     * @param tolerance     The error in which to stop the training iterations.
     * @return The error of the last training iteration.
     */
    double fit(Matrix[] input, Matrix[] output, double maxIterations, double tolerance);

    /**
     * Fit the machine learning algorithm to an input and output data set.
     *
     * @param input  The input of the algorithm.
     * @param output The desired output of the algorithm in the same order as the input.
     * @return The error of the last training iteration.
     */
    double fit(Matrix[] input, Matrix[] output);

    /**
     * Calculates the percent of predictions that are correct.
     *
     * @param input  The input of the machine learning algorithm.
     * @param output The desired output of the algorithm in the same order as the input.
     * @return The accuracy of the predictions from [0, 1].
     */
    double accuracy(Matrix[] input, Matrix[] output);

    /**
     * Calculates the percent of predictions that are correct.
     *
     * @param input  The input of the machine learning algorithm.
     * @param output The desired output of the algorithm in the same order as the input.
     * @param argmax Determines if the accuracy should be calculated based on which output prediction was greatest.
     * @return The accuracy of the predictions from [0, 1].
     */
    double accuracy(Matrix[] input, Matrix[] output, boolean argmax);

}
