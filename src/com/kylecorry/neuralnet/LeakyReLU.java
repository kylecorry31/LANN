package com.kylecorry.neuralnet;

public class LeakyReLU extends Activation {

	@Override
	public double activate(double x) {
		return x > 0 ? x : 0.01 * x;
	}

	@Override
	public double derivitive(double x) {
		// TODO Auto-generated method stub
		return x > 0 ? 1 : 0.01;
	}

}
