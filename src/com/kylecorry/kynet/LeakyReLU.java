package com.kylecorry.kynet;

public class LeakyReLU extends Activation {

	@Override
	public double activate(double x) {
		return x > 0 ? x : 0.01 * x;
	}

	@Override
	public double derivative(double x) {
		// TODO Auto-generated method stub
		return x > 0 ? 1 : 0.01;
	}

}
