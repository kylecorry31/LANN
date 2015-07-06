package com.kylecorry.neuralnet;

public class Binary extends Activation {

	@Override
	public double activate(double x) {
		return x > 0 ? 1 : 0;
	}

	@Override
	public double derivative(double x) {
		return 0;
	}

}
