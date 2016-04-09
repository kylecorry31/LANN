package com.kylecorry.lann.activation;

public class Sigmoid extends Activation {

	@Override
	public double activate(double x) {
		return 1d / (1 + Math.pow(Math.E, -x));
	}

	@Override
	public double derivative(double x) {
//		return -Math.exp(-x) / Math.pow(1+Math.exp(-x), 2);
		return activate(x) * (1 - activate(x));
	}

}
