#include <random>
#include <algorithm>
#include "NeuralNetwork.hpp"

namespace nn
{
	// NeuronNetwork
	NeuralNetwork::NeuralNetwork(unsigned int inputNumber, unsigned int outputNumber, const std::vector<unsigned int>& hiddenLayersDefinition)
		: _layers(hiddenLayersDefinition.size() + 1)
	{
		for (unsigned int i = 0; i < hiddenLayersDefinition.size(); ++i)
		{
			this->_layers[i] = new NeuronLayer(inputNumber, hiddenLayersDefinition[i]);
			inputNumber = hiddenLayersDefinition[i];
		}
		this->_layers.back() = new NeuronLayer(inputNumber, outputNumber);
	}
	
	NeuralNetwork::~NeuralNetwork()
	{
		for (unsigned int i = 0; i < this->_layers.size(); ++i)
		{
			delete this->_layers[i];
		}
	}
	
	NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
		: _layers(other._layers.size())
	{
		for (unsigned int i = 0; i < this->_layers.size(); ++i)
		{
			this->_layers[i] = new NeuronLayer(*(other._layers[i]));
		}
	}
	
	NeuralNetwork::NeuralNetwork(const NeuralNetwork&& other)
	{
		for (unsigned int i = 0; i < this->_layers.size(); ++i)
		{
			this->_layers[i] = new NeuronLayer(*(other._layers[i]));
		}
	}

	NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& other)
	{
		if (this != &other)
		{
			if (this->_layers.size() != other._layers.size())
				throw 42; // TODO real exception
			for (unsigned int i = 0; i < this->_layers.size(); ++i)
			{
				*(this->_layers[i]) = *(other._layers[i]);
			}
		}
		return *this;
	}
	
	std::vector<std::pair<bool, double> > NeuralNetwork::update(const std::vector<bool>& userInput) const
	{
		if (userInput.size() != this->_layers[0]->getInputNumber())
			throw "ERROR PA KOOL";

		std::vector<std::pair<bool, double> > input(userInput.size());
		std::vector<std::pair<bool, double> > output;
		
		for (unsigned int i = 0; i < userInput.size(); ++i)
		{
			input[i] = std::pair<bool, double>(userInput[i], userInput[i]);
		}

		for (unsigned int i = 0; i < this->_layers.size(); ++i)			// For each layer
		{
			const NeuronLayer& currentLayer(*this->_layers[i]);

			output.clear();
			output.reserve(currentLayer.size());
			
			for (unsigned int j = 0; j < currentLayer.size(); ++j)		// For each neuron
			{
				const Neuron& currentNeuron(currentLayer[j]);
				
				double activation = 0;
				
				for (unsigned int k = 0; k < input.size(); ++k)			// For each input
				{
					activation += input[k].first * currentNeuron.getWeight(k);
				}
				
				output[j] = std::pair<bool, double>(activation >= currentNeuron.getThreshold(), std::max<double>(0, (activation - currentNeuron.getThreshold()) / (1 - currentNeuron.getThreshold())));
				
//				activation 																=> activation
//				activation / threshold												=> pourcentage d'activation
//				max(0, (activation - threshold) / (1 - threshold))	=> pourcentage d'activation manquant
			}
			
			input = output;
		}
		return output;
	}
	
	void NeuralNetwork::save(const std::string& filename) const
	{
		throw "TODO " + filename;
	}

	void NeuralNetwork::load(const std::string& filename)
	{
		throw "TODO " + filename;
	}
	
	// NeuronLayer
	NeuronLayer::NeuronLayer(unsigned int inputNumber, unsigned int neuronNumber)
		: _neurons(neuronNumber), _inputNumber(inputNumber)
	{
		for (unsigned int i = 0; i < neuronNumber; ++i)
		{
			this->_neurons[i] = new Neuron(inputNumber);
		}
	}
	
	NeuronLayer::~NeuronLayer()
	{
		for (unsigned int i = 0; i < this->_neurons.size(); ++i)
		{
			delete this->_neurons[i];
		}
	}
	
	NeuronLayer::NeuronLayer(const NeuronLayer& other)
		: _neurons(other._neurons.size()), _inputNumber(other._inputNumber)
	{
		for (unsigned int i = 0; i < this->_neurons.size(); ++i)
		{
			this->_neurons[i] = new Neuron(*(other._neurons[i]));
		}
	}
	
	NeuronLayer::NeuronLayer(const NeuronLayer&& other)
		: _neurons(other._neurons.size()), _inputNumber(other._inputNumber)
	{
		for (unsigned int i = 0; i < this->_neurons.size(); ++i)
		{
			this->_neurons[i] = new Neuron(*(other._neurons[i]));
		}
	}
	
	NeuronLayer& NeuronLayer::operator=(const NeuronLayer& other)
	{
		if (this != &other)
		{
			if (this->_neurons.size() != other._neurons.size())
				throw 42; // TODO real exception
			for (unsigned int i = 0; i < this->_neurons.size(); ++i)
			{
				*(this->_neurons[i]) = *(other._neurons[i]);
			}
			this->_inputNumber = other._inputNumber;
		}
		return *this;
	}

	Neuron& NeuronLayer::operator[](unsigned int idx)
	{
		if (idx >= this->_neurons.size())
			throw 42; // TODO real exception
		return *(this->_neurons[idx]);
	}
	
	const Neuron& NeuronLayer::operator[](unsigned int idx) const
	{
		if (idx >= this->_neurons.size())
			throw 42; // TODO real exception
		return *(this->_neurons[idx]);
	}
	
	unsigned int NeuronLayer::size() const
	{
		return this->_neurons.size();
	}
	
	unsigned int NeuronLayer::getInputNumber() const
	{
		return this->_inputNumber;
	}
	
	// Neuron
	Neuron::Neuron(unsigned int inputNumber)
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
		this->_threshold = uniform_dist(e1);
	}
	
	Neuron::~Neuron()
	{ }
	
	Neuron::Neuron(const Neuron& other)
		: _weights(other._weights), _threshold(other._threshold)
	{}

	Neuron::Neuron(const Neuron&& other)
		: _weights(other._weights), _threshold(other._threshold)
	{}

	Neuron&	Neuron::operator=(const Neuron& other)
	{
		if (this != &other)
		{
			this->_weights = other._weights;
			this->_threshold = other._threshold;
		}
		return *this;
	}

	double& Neuron::getWeight(unsigned int idx)
	{
		if (idx >= this->_weights.size())
			throw 42; // TODO True exception
		return this->_weights[idx];
	}
	
	double Neuron::getWeight(unsigned int idx) const
	{
		if (idx >= this->_weights.size())
			throw 42; // TODO True exception
		return this->_weights[idx];
	}
	
	double Neuron::getThreshold() const
	{
		return this->_threshold;
	}
	
	void Neuron::setThreshold(double value)
	{
		this->_threshold = value;
	}
}