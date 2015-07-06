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

	public static double varience(double[] array) {
		double average = average(array);
		double[] newArr = array.clone();
		for (int i = 0; i < newArr.length; i++) {
			newArr[i] = Math.pow(newArr[i] - average, 2);
		}
		return sum(newArr);
	}

	public static double average(double[] array) {
		return sum(array) / array.length;
	}
}
