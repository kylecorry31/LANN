package com.kylecorry.neuralnet;

public class Utils {
	public static double sum(double input[]) {
		double total = 0;
		for (double item : input) {
			total += item;
		}
		return total;
	}

	public static int sum(int input[]) {
		int total = 0;
		for (int item : input) {
			total += item;
		}
		return total;
	}
}
