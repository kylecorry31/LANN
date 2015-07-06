package com.kylecorry.kynet;

public class Tanh extends Activation {

	@Override
	public double activate(double x) {
		return Math.tanh(x);
	}

	@Override
	public double derivative(double x) {
		// TODO Auto-generated method stub
		return 1 - Math.pow(Math.tanh(x), 2);
	}

}
