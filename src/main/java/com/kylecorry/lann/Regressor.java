package com.kylecorry.lann;

import com.kylecorry.matrix.Matrix;

public class Regressor implements IRegressor {

    private MachineLearningAlgorithm machineLearningAlgorithm;

    /**
     * Creates a regressor. Assumes the machine learning algorithm returns a matrix with a single value.
     * @param machineLearningAlgorithm A machine learning algorithm, which returns a matrix with a single value.
     */
    public Regressor(MachineLearningAlgorithm machineLearningAlgorithm) {
        this.machineLearningAlgorithm = machineLearningAlgorithm;
    }

    @Override
    public double predict(Matrix input) {
        Matrix prediction = machineLearningAlgorithm.predict(input);
        return prediction.get(0, 0);
    }
}
