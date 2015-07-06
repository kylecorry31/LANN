package com.kylecorry.neuralnet;

public class Softplus extends Activation {

	@Override
	public double activate(double x) {
		return Math.log(1 + Math.pow(Math.E, x));
	}

	@Override
	public double derivative(double x) {
		return 1d / (1 + Math.pow(Math.E, -x));
	}

}
