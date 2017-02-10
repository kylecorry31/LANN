package com.kylecorry.lann.activation;

public class Linear implements Activation {

	@Override
	public double activate(double x) {

		return x;
	}

	@Override
	public double derivative(double x) {
		return 1;
	}

}
