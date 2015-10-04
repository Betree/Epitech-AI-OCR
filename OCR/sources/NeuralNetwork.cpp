#include <random>
#include "NeuralNetwork.hpp"

namespace nn
{
	// Neuron
	Neuron::Neuron(int inputNumber)
		: _weights(inputNumber)
	{
		std::random_device rd;
	 
		// Choose a random mean between 1 and 6
		std::default_random_engine e1(rd());
		std::uniform_real_distribution<double> uniform_dist(0, 1);

		for (unsigned int i = 0; i < inputNumber; ++i)
		{
			this->_weights[i] = uniform_dist(e1);
		}
	}
	
	Neuron::~Neuron()
	{ }
	
	Neuron::Neuron(const Neuron& other)
		: _weights(other._weights)
	{}
	
	Neuron::Neuron(const Neuron&& other)
		: _weights(other._weights)
	{}

	Neuron&	Neuron::operator=(const Neuron& other)
	{
		if (this != &other)
		{
			this->_weights = other._weights;
		}
	}

	double& Neuron::operator[](unsigned int idx)
	{
		if (idx >= this->_weights.size())
			throw 42; // TODO True exception
		return this->_weights[idx];
	}
	
	double Neuron::operator[](unsigned int idx) const
	{
		if (idx >= this->_weights.size())
			throw 42; // TODO True exception
		return this->_weights[idx];
	}
}