package com.kylecorry.lann;

import com.kylecorry.matrix.Matrix;

/**
 * Created by Kyle on 6/17/2017.
 */
public class Classifier<T> implements IClassifier<T> {

    private MachineLearningAlgorithm machineLearningAlgorithm;
    private T[] labels;

    public Classifier(MachineLearningAlgorithm machineLearningAlgorithm, T[] labels) {
        this.machineLearningAlgorithm = machineLearningAlgorithm;
        this.labels = labels;
    }

    @Override
    public Classification<T> classify(Matrix input) {
        final Matrix prediction = machineLearningAlgorithm.predict(input);
        final int labelPos = NeuralNetworkPredictionAnalyzer.argMax(prediction);
        final T label = labels[labelPos];
        final double confidence = prediction.get(labelPos, 0);
        return new Classification<>(label, confidence);
    }
}
