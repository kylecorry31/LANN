package com.kylecorry.neuralnet;

public class Tanh extends Activation {

	@Override
	public double activate(double x) {
		return Math.tanh(x);
	}

	@Override
	public double derivitive(double x) {
		// TODO Auto-generated method stub
		return 1 - Math.pow(Math.tanh(x), 2);
	}

}
