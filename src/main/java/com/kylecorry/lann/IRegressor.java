package com.kylecorry.lann;

import com.kylecorry.matrix.Matrix;

public interface IRegressor {

    /**
     * Predict an input given an output.
     * @param input The input to the regressor.
     * @return The prediction of the regression algorithm.
     */
    double predict(Matrix input);
}
