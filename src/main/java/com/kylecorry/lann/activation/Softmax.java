package com.kylecorry.lann.activation;

public class Softmax implements Activation {

	@Override
	public double activate(double x) {
		return Math.pow(Math.E, x);
	}

	@Override
	public double derivative(double x) {
		return Math.pow(Math.E, x) - x;
	}

}
