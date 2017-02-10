package com.kylecorry.lann.activation;

public class Tanh implements Activation {

	@Override
	public double activate(double x) {
		return Math.tanh(x);
	}

	@Override
	public double derivative(double x) {
		return 1 - Math.pow(Math.tanh(x), 2);
	}

}
