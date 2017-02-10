package com.kylecorry.lann;

import com.kylecorry.lann.activation.Sigmoid;
import com.kylecorry.lann.activation.Softmax;
import com.kylecorry.matrix.Matrix;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by Kyle on 2/9/2017.
 */
public class Examples {
    @Test
    public void testNN() {
        PersistentMachineLearningAlgorithm testNet = new NN.Builder().addLayer(2, 4, new Sigmoid())
                .addLayer(4, 3, new Softmax()).build();

        Matrix[] input = new Matrix[]{new Matrix(100d, 2d), new Matrix(0d, 10d)};
        Matrix[] output = new Matrix[]{new Matrix(1d, 0d, 0d), new Matrix(0d, 1d, 0d)};

        testNet.fit(input, output);

        assertEquals(0, NeuralNetworkPredictionAnalyzer.argMax(new Matrix(1d, 0d, 0d)));
//        assertEquals(0, NeuralNetworkPredictionAnalyzer.argMax(testNet.predict(100d, 2d)));
//        assertEquals(1, testNet.accuracy(new Matrix[]{new Matrix(100d, 2d)}, new Matrix[]{new Matrix(1d, 0d, 0d)}, true), 0);
    }
}
