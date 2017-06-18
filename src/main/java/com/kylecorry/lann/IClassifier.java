package com.kylecorry.lann;

import com.kylecorry.matrix.Matrix;

/**
 * Created by Kyle on 6/17/2017.
 */
public interface IClassifier<T> {

    /**
     * Classify the input.
     *
     * @param input The input of the machine learning algorithm.
     * @return The classification.
     */
    Classification<T> classify(Matrix input);

    class Classification<T> {
        private T classification;
        private double confidence;

        protected Classification(T classification, double confidence) {
            this.classification = classification;
            this.confidence = confidence;
        }

        public T getClassification() {
            return classification;
        }

        public double getConfidence() {
            return confidence;
        }
    }


}
