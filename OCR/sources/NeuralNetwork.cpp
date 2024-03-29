#include <random>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include "NeuralNetwork.hpp"

using namespace std;

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

	NeuralNetwork::NeuralNetwork()
	{
	}
	
	NeuralNetwork::~NeuralNetwork()
	{
		this->cleanup();
	}
	
	NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
		: _layers(other._layers.size())
	{
		for (unsigned int i = 0; i < this->_layers.size(); ++i)
		{
			this->_layers[i] = new NeuronLayer(*(other._layers[i]));
		}
	}
	
	NeuralNetwork::NeuralNetwork(NeuralNetwork&& other)
		: _layers(std::move(other._layers))
	{ }

	NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& other)
	{
		if (this != &other)
		{
			this->cleanup();
			for (size_t i = 0; i < other._layers.size(); i++)
			{
				this->_layers.push_back(new NeuronLayer(*other._layers[i]));
			}
		}
		return *this;
	}

	NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork&& other)
	{
		this->cleanup();
		this->_layers = std::move(other._layers);
		return *this;
	}

	unsigned int NeuralNetwork::size() const
	{
		return this->_layers.size();
	}

	NeuronLayer& NeuralNetwork::operator[](unsigned int idx)
	{
		if (idx >= this->_layers.size())
			throw runtime_error("OUT OF RANGE");
		return *this->_layers[idx];
	}

	const NeuronLayer& NeuralNetwork::operator[](unsigned int idx) const
	{
		if (idx >= this->_layers.size())
			throw runtime_error("OUT OF RANGE");
		return *this->_layers[idx];
	}

	double NeuralNetwork::sigmoid_prime(double value)
	{
		return std::exp(value) / std::pow(std::exp(value) + 1, 2);
	}

	double NeuralNetwork::sigmoid(double value)
	{
		return (1.0 / (1.0 + std::exp(-value)));
	}

	NeuralFeed NeuralNetwork::update(NeuralFeed input) const
	{
		if (input.size() != this->_layers[0]->getInputNumber())
			throw runtime_error("input size mismatch");

		std::vector<double> output;

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
					activation += input[k] * currentNeuron.getWeight(k);
				}
				
				activation += currentNeuron.getBias();
				output.push_back(sigmoid(activation));
			}
			
			input = output;
		}
		return output;
	}
	
	bool NeuralNetwork::save(const std::string& filename) const
	{
		std::ofstream file(filename, std::ios_base::out | std::ios_base::trunc);
		bool sent;

		if ((sent = !!file))
			for (size_t i = 0; i < this->_layers.size(); i++)
			{
				const NeuronLayer& layer(*this->_layers[i]);

				file << layer.size() << ' ' << (layer.size() > 0 ? layer[0].size() : '0');

				for (size_t j = 0; j < layer.size(); j++)
				{
					const Neuron& neuron(layer[j]);

					file << '\t' << neuron.getBias();
					for (size_t k = 0; k < neuron.size(); k++)
					{
						file << ' ' << neuron.getWeight(k);
					}
				}
				file << std::endl;
			}
		if (!sent)
			file.close();
		return sent;
	}

	bool NeuralNetwork::load(const std::string& filename)
	{
		bool sent = true;
		this->cleanup();
		std::ifstream file(filename);
		std::string line;

		if (!file)
			sent = false;
		while (sent && std::getline(file, line))
		{
			std::istringstream stream(line);
			unsigned int inputNumber;
			unsigned int neuronNumber;

			stream >> neuronNumber >> inputNumber;

			if (!stream)
				sent = false;
			else
			{
				this->_layers.push_back(new NeuronLayer(inputNumber, neuronNumber));
				NeuronLayer& layer(*this->_layers.back());

				for (size_t i = 0; sent && i < neuronNumber; i++)
				{
					Neuron& neuron = layer[i];
					double bias;
					double weight;

					if (!(stream >> bias))
						sent = false;
					else
					{
						neuron.setBias(bias);
						for (size_t j = 0; sent && j < inputNumber; j++)
						{
							if (!(stream >> weight))
								sent = false;
							else
							{
								neuron.setWeight(j, weight);
							}
						}
					}
				}
			}
		}
		if (!sent)
			this->cleanup();
		return sent;
	}

	void NeuralNetwork::cleanup()
	{
		for (unsigned int i = 0; i < this->_layers.size(); ++i)
		{
			delete this->_layers[i];
		}
		this->_layers.clear();
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
				throw runtime_error("OUT OF RANGE");
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
			throw runtime_error("OUT OF RANGE");
		return *(this->_neurons[idx]);
	}
	
	const Neuron& NeuronLayer::operator[](unsigned int idx) const
	{
		if (idx >= this->_neurons.size())
			throw runtime_error("OUT OF RANGE");
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
		std::uniform_real_distribution<double> uniform_dist(-1, 1);

		for (unsigned int i = 0; i < inputNumber; ++i)
		{
			this->_weights[i] = uniform_dist(e1);
		}
		this->_bias = uniform_dist(e1);
	}
	
	Neuron::~Neuron()
	{ }
	
	Neuron::Neuron(const Neuron& other)
		: _weights(other._weights), _bias(other._bias)
	{}

	Neuron::Neuron(const Neuron&& other)
		: _weights(other._weights), _bias(other._bias)
	{}

	Neuron&	Neuron::operator=(const Neuron& other)
	{
		if (this != &other)
		{
			this->_weights = other._weights;
			this->_bias = other._bias;
		}
		return *this;
	}

	double& Neuron::getWeight(unsigned int idx)
	{
		if (idx >= this->_weights.size())
			throw runtime_error("OUT OF RANGE");
		return this->_weights[idx];
	}
	
	double Neuron::getWeight(unsigned int idx) const
	{
		if (idx >= this->_weights.size())
			throw runtime_error("OUT OF RANGE");
		return this->_weights[idx];
	}

	void Neuron::setWeight(unsigned int idx, double value)
	{
		if (idx >= this->_weights.size())
			throw runtime_error("OUT OF RANGE");
		this->_weights[idx] = value;
	}

	size_t Neuron::size() const
	{
		return this->_weights.size();
	}
	
	double Neuron::getBias() const
	{
		return this->_bias;
	}
	
	void Neuron::setBias(double value)
	{
		this->_bias = value;
	}
}