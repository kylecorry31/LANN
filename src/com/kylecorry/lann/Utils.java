package com.kylecorry.lann;

class Utils {
	/**
	 * Calculates the sum of an array.
	 * 
	 * @param input
	 *            An array of doubles.
	 * @return The sum of the values in the array.
	 */
	static double sum(double input[]) {
		double total = 0;
		for (double item : input) {
			total += item;
		}
		return total;
	}

	/**
	 * Calculates the sum of an array.
	 * 
	 * @param input
	 *            An array of integers.
	 * @return The sum of the values in the array.
	 */
	static int sum(int input[]) {
		int total = 0;
		for (int item : input) {
			total += item;
		}
		return total;
	}

	/**
	 * Calculates the variance of an array.
	 * 
	 * @param array
	 *            An array of doubles.
	 * @return The variance of the array.
	 */
	static double variance(double[] array) {
		double average = average(array);
		double[] newArr = array.clone();
		for (int i = 0; i < newArr.length; i++) {
			newArr[i] = Math.pow(newArr[i] - average, 2);
		}
		return sum(newArr);
	}

	/**
	 * Calculates the average of an array.
	 * 
	 * @param array
	 *            An array of doubles.
	 * @return The average of the array.
	 */
	static double average(double[] array) {
		return sum(array) / array.length;
	}
}
