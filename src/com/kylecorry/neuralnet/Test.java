package com.kylecorry.neuralnet;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class Test {

	public static void main(String[] args) {
		Network net = new Network(new int[] { 2, 4, 1 }, new Activation[] {
				new Linear(), new Tanh(), new Sigmoid() });
		for (double output : net.activate(new double[] { 1, 1 })) {
			System.out.println(output);
		}

		for (double output : net.activate(new double[] { 1, 0 })) {
			System.out.println(output);
		}

		for (double output : net.activate(new double[] { 0, 1 })) {
			System.out.println(output);
		}

		for (double output : net.activate(new double[] { 0, 0 })) {
			System.out.println(output);
		}
		System.out.println("____________________________________");
		// for (int i = 0; i < 5000; i++)
		// net.train(
		// new double[][] { { 1, 1 }, { 1, 0 }, { 0, 0 }, { 0, 1 } },
		// new double[][] { { 0 }, { 1 }, { 0 }, { 1 } });

		net.weightsFromFile("weights.csv");
		for (double output : net.activate(new double[] { 1, 1 })) {
			System.out.println(output);
		}

		for (double output : net.activate(new double[] { 1, 0 })) {
			System.out.println(output);
		}

		for (double output : net.activate(new double[] { 0, 1 })) {
			System.out.println(output);
		}

		for (double output : net.activate(new double[] { 0, 0 })) {
			System.out.println(output);
		}

	}

}
