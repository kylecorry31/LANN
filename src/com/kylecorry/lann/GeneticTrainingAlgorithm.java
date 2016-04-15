package com.kylecorry.lann;

import java.util.HashMap;

import com.kylecorry.lann.activation.Linear;
import com.kylecorry.lann.activation.Sigmoid;
import com.kylecorry.lann.activation.Softmax;
import com.kylecorry.matrix.Matrix;

public class GeneticTrainingAlgorithm {

	private HashMap<NeuralNetwork, Double> genes;
	private double learningRate;
	private double mutationRate;
	private boolean keepBest;

//	public static void main(String[] args) {
//		GeneticTrainingAlgorithm gta = new GeneticTrainingAlgorithm(new int[] { 2, 3, 2 }, 15, 0.1, 0.75, true);
//		HashMap<NeuralNetwork, Double> g = gta.getGenes();
//		for (int i = 0; i < 1000; i++) {
//			int count = 0;
//			for (NeuralNetwork net : g.keySet()) {
//				gta.setFitness(net, net.squaredError(new Matrix(new double[][] { { 0.5 }, { 0.2 } }),
//						new Matrix(new double[][] { { 1 }, { 0 } })));
//				System.out.println(count + " " + gta.getFitness(net));
//				count++;
//			}
//			System.out.println();
//			gta.evolve(false);
//		}
//		System.out.println();
//		NeuralNetwork net = new NeuralNetwork.Builder().addLayer(2, 3, new Linear()).addLayer(3, 2, new Softmax())
//				.build();
//		for (int i = 0; i < 1000; i++)
//			net.train(new Matrix(new double[][] { { 0.5, 0.2 } }), new Matrix(new double[][] { { 0.4, 0.2 } }), 0.01);
//		System.out.println(net.squaredError(new Matrix(new double[][] { { 0.5 }, { 0.2 } }),
//				new Matrix(new double[][] { { 1 }, { 0 } })));
//
//	}

	public HashMap<NeuralNetwork, Double> getGenes() {
		return genes;
	}

	public GeneticTrainingAlgorithm(int[] topology, int numGenes, double learningRate, double mutation,
			boolean classify, boolean keepBest) {
		genes = new HashMap<NeuralNetwork, Double>();
		for (int i = 0; i < numGenes; i++) {
			NeuralNetwork.Builder netBuilder = new NeuralNetwork.Builder();
			for (int t = 1; t < topology.length; t++) {
				if (t == 1)
					netBuilder.addLayer(topology[t - 1], topology[t], new Linear());
				else if (t == topology.length - 1 && classify)
					netBuilder.addLayer(topology[t - 1], topology[t], new Softmax());
				else
					netBuilder.addLayer(topology[t - 1], topology[t], new Sigmoid());
			}
			genes.put(netBuilder.build(), 0.0);
			this.learningRate = learningRate;
			this.mutationRate = mutation;
			this.keepBest = keepBest;
		}
	}

	public void load(String filename){
		NeuralNetwork net = genes.keySet().iterator().next();
		net.load(filename);
		setFitness(net, 100000);
		evolve(true);
	}
	
	public void setFitness(NeuralNetwork net, double score) {
		genes.put(net, score);
	}

	public double getFitness(NeuralNetwork net) {
		return genes.get(net);
	}

	public void evolve(boolean max) {
		NeuralNetwork best = best(max);
		int count = 0;
		for (NeuralNetwork net : genes.keySet()) {
			if(count == 0 && keepBest){
				net = best;
				continue;
			}
			for (int i = 0; i < net.size(); i++) {
				if (Math.random() < mutationRate)
					net.setWeights(i, best.getWeights(i).add((Math.random() * 2 - 1) * learningRate));
				else
					net.setWeights(i, best.getWeights(i));
			}
			genes.put(net, 0.0);
			count++;
		}
	}

	private NeuralNetwork max() {
		double max = -Double.MAX_VALUE;
		NeuralNetwork maxNet = null;
		for (NeuralNetwork net : genes.keySet()) {
			double score = getFitness(net);
			if (score > max) {
				max = score;
				maxNet = net;
			}
		}
		return maxNet;
	}

	private NeuralNetwork min() {
		double max = Double.MAX_VALUE;
		NeuralNetwork maxNet = null;
		for (NeuralNetwork net : genes.keySet()) {
			double score = getFitness(net);
			if (score < max) {
				max = score;
				maxNet = net;
			}
		}
		return maxNet;
	}

	public NeuralNetwork best(boolean max) {
		return max ? max() : min();
	}

	public double bestFitness(boolean max) {
		return getFitness(best(max));
	}

}
