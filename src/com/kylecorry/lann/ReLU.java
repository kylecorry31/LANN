package com.kylecorry.lann;

public class ReLU extends Activation {

	@Override
	public double activate(double x) {
		return Math.max(0, x);
	}

	@Override
	public double derivative(double x) {
		return x > 0 ? 1 : 0;
	}

}
