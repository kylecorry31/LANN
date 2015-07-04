package com.kylecorry.neuralnet;

public class Linear extends Activation {

	@Override
	public double activate(double x) {

		return x;
	}

	@Override
	public double derivitive(double x) {
		// TODO Auto-generated method stub
		return 1;
	}

}
