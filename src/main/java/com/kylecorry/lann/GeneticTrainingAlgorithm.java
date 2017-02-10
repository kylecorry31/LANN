package com.kylecorry.lann;

import com.kylecorry.lann.activation.Linear;
import com.kylecorry.lann.activation.Sigmoid;
import com.kylecorry.lann.activation.Softmax;

import java.util.HashMap;

@Deprecated
public class GeneticTrainingAlgorithm {

	private HashMap<NeuralNetwork, Double> genes;
	private double learningRate;
	private double mutationRate;
	private boolean keepBest;

	/**
	 * Get a HashMap linking the neural networks to their fitness.
	 * 
	 * @return A HashMap<NeuralNetwork, Double> linking the nets with their
	 *         fitness.
	 */
	public HashMap<NeuralNetwork, Double> getGenes() {
		return genes;
	}

	/**
	 * Train the neural networks using a version of a genetic training
	 * algorithm.
	 * 
	 * @param topology
	 *            The topology of the networks in this format int[]{input,
	 *            hidden1, hidden2, hiddenN, output}
	 * @param numGenes
	 *            The number of different genes per generation.
	 * @param learningRate
	 *            The amount a mutation affects the genes of the next
	 *            generation.
	 * @param mutation
	 *            The percent chance of a mutation occurring: a double from 0..1
	 *            inclusive
	 * @param classify
	 *            A boolean flag specifying whether this is a classification
	 *            problem.
	 * @param keepBest
	 *            A boolean flag specifying whether to keep the gene with the
	 *            best fitness in the next generation without mutation.
	 */
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

	/**
	 * Load a gene pool from a file.
	 * 
	 * @param filename
	 *            The filename of the file containing the weight values for a
	 *            neural net.
	 */
	public void load(String filename) {
		NeuralNetwork net = genes.keySet().iterator().next();
		net.load(filename);
		setFitness(net, 100000);
		evolve(true);
	}

	/**
	 * Set the fitness of a neural network.
	 * 
	 * @param net
	 *            The neural network being evaluated.
	 * @param score
	 *            The fitness of the neural network.
	 */
	public void setFitness(NeuralNetwork net, double score) {
		genes.put(net, score);
	}

	/**
	 * Get the fitness of a neural network.
	 * 
	 * @param net
	 *            The neural network being evaluated.
	 * @return The fitness of the neural network.
	 */
	public double getFitness(NeuralNetwork net) {
		return genes.get(net);
	}

	/**
	 * Evolve the neural networks to create the next generation.
	 * 
	 * @param max
	 *            A boolean flag of whether to maximize or minimize the fitness.
	 *            Max is true, min is false.
	 */
	public void evolve(boolean max) {
		NeuralNetwork best = best(max);
		int count = 0;
		for (NeuralNetwork net : genes.keySet()) {
			if (count == 0 && keepBest) {
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

	/**
	 * Get the neural network with the highest fitness.
	 * 
	 * @return The neural network with the highest fitness.
	 */
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

	/**
	 * Get the neural network with the lowest fitness.
	 * 
	 * @return The neural network with the lowest fitness.
	 */
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

	/**
	 * Get the best neural network based on its fitness.
	 * 
	 * @param max
	 *            A boolean flag of whether a higher fitness is better than a
	 *            lower fitness. true for max, false for min.
	 * @return The best neural network based on its fitness.
	 */
	public NeuralNetwork best(boolean max) {
		return max ? max() : min();
	}

	/**
	 * Get the best fitness in the generation.
	 * 
	 * @param max
	 *            A boolean flag of whether a higher fitness is better than a
	 *            lower fitness. true for max, false for min.
	 * @return The fitness of the best neural network.
	 */
	public double bestFitness(boolean max) {
		return getFitness(best(max));
	}

}
