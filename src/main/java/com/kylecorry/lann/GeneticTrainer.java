package com.kylecorry.lann;

import com.kylecorry.lann.activation.Linear;
import com.kylecorry.lann.activation.Sigmoid;
import com.kylecorry.lann.activation.Softmax;

import java.util.ArrayList;
import java.util.List;

public class GeneticTrainer {
	private List<Gene> genes;
	private double learningRate;
	private double mutationRate;
	private boolean keepBest;

	public GeneticTrainer(int[] topology, int numGenes, double learningRate, double mutation, boolean classify,
			boolean keepBest) {
		genes = new ArrayList<Gene>();
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
			genes.add(new GeneticTrainer.Gene(netBuilder.build(), 0.0));
			this.learningRate = learningRate;
			this.mutationRate = mutation;
			this.keepBest = keepBest;
		}
	}

	public static class Gene {
		private NeuralNetwork network;
		private double fitness;

		public Gene(NeuralNetwork nn, double fit) {
			fitness = fit;
			network = nn;
		}

		public NeuralNetwork getNeuralNetwork() {
			return network;
		}

		public double getFitness() {
			return fitness;
		}
	}
}
