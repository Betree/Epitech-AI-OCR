#include <algorithm>
#include "Trainer.hpp"
#include "indexed_vector.hpp"
#include<iostream> // TODO remove

namespace nn
{
	Trainer::Trainer(NeuralNetwork* network)
		: _network(network), _miniBatchSize(100), _miniBatch()
	{
		this->_miniBatch.reserve(100);
	}

	Trainer::~Trainer()
	{
	}

	Trainer::Trainer(const Trainer& other)
		: _network(other._network), _miniBatchSize(other._miniBatchSize)
	{

	}

	Trainer & Trainer::operator=(const Trainer& other)
	{
		if (this != &other)
		{
			this->_network = other._network;
			this->_miniBatchSize = other._miniBatchSize;
			this->_miniBatch = other._miniBatch;
		}
		return *this;
	}

	unsigned int Trainer::getMiniBatchSize() const
	{
		return this->_miniBatchSize;
	}

	void Trainer::setMiniBatchSize(unsigned int value)
	{
		this->_miniBatchSize = value;
	}

	void Trainer::train(const Trainer::Epoch& in)
	{
		std::vector<InputOutputPair> epoch(in);

		this->_miniBatch.clear();
		std::random_shuffle(epoch.begin(), epoch.end());
		for (size_t i = 0; i < epoch.size(); i++)
		{
			this->feed(epoch[i]);
		}
		this->flush();
	}

	void Trainer::feed(const InputOutputPair& input)
	{
		this->_miniBatch.push_back(input);
		if (this->_miniBatch.size() >= this->_miniBatchSize)
			this->flush();
	}
	
	static const unsigned int eta = 1;

	static void backpropagation(NeuralNetwork& network, const std::vector<nn::Trainer::InputOutputPair>& batch)
	{
		const unsigned int L = network.size() + 1;
		std::vector<indexed_vector<1, indexed_vector<0, double> > > as(batch.size());
		indexed_vector<2, indexed_vector<0, double>> z(L - 1);
		std::vector<indexed_vector<2, indexed_vector<0, double> > > deltas(batch.size());

		for (size_t i = 0; i < batch.size(); ++i)
		{
			const nn::Trainer::InputOutputPair& test(batch[i]);
			deltas[i] = indexed_vector<2, indexed_vector<0, double> >(L - 1);
			indexed_vector<2, indexed_vector<0, double>>& delta(deltas[i]);

			as[i] = indexed_vector<1, indexed_vector<0, double> >(L);
			indexed_vector<1, indexed_vector<0, double>>& a(as[i]);

			// Feed forward
			a[1] = test.first;
			for (size_t l = 2; l <= L; l++)
			{
				nn::NeuronLayer& layer(network[l - 2]);

				z[l] = indexed_vector<0, double>(layer.size());
				a[l] = indexed_vector<0, double>(layer.size());

				for (size_t j = 0; j < layer.size(); j++)
				{
					nn::Neuron& neuron(layer[j]);

					z[l][j] = 0;
					for (size_t k = 0; k < a[l - 1].size(); k++)
					{
						z[l][j] += a[l - 1][k] * neuron.getWeight(k);
					}
					z[l][j] += neuron.getBias();
					a[l][j] = network.sigmoid(z[l][j]);
				}
			}

			// Output error
			delta[L] = indexed_vector<0, double>(a[L].size());
			for (size_t j = 0; j < a[L].size(); j++)
			{
				delta[L][j] = (a[L][j] - test.second[j]) * network.sigmoid_prime(z[L][j]);
			}

			// Backpropagation of the error
			for (size_t l = L - 1; l >= 2; l--)
			{
				nn::NeuronLayer& layerp1(network[l - 2 + 1]);

				delta[l] = indexed_vector<0, double>(a[l].size());
				for (size_t k = 0; k < a[l].size(); k++)
				{
					delta[l][k] = 0;
					for (size_t j = 0; j < a[l + 1].size(); j++)
					{
						delta[l][k] += layerp1[j].getWeight(k) * delta[l + 1][j];
					}
					delta[l][k] *= network.sigmoid_prime(z[l][k]);
				}
			}
		}

		for (size_t l = L; l >= 2; l--)
		{
			nn::NeuronLayer& layer(network[l - 2]);

			for (size_t j = 0; j < layer.size(); j++)
			{
				nn::Neuron& neuron(layer[j]);

				double deltaBias = 0;
				for (size_t x = 0; x < deltas.size(); x++)
				{
					deltaBias += deltas[x][l][j];
				}
				for (size_t w = 0; w < neuron.size(); w++)
				{
					double deltaWeight = 0;

					for (size_t x = 0; x < batch.size(); x++)
					{
						deltaWeight += deltas[x][l][j] * as[x][l - 1][w];
					}
					neuron.setWeight(w, neuron.getWeight(w) - ((double)eta / (double)batch.size()) * deltaWeight);
				}
				neuron.setBias(neuron.getBias() - ((double)eta / (double)batch.size()) * deltaBias);
			}
		}
	}

	void Trainer::flush()
	{
		if (!this->_miniBatch.empty())
		{
			backpropagation(*this->_network, this->_miniBatch);
			this->_miniBatch.clear();
		}
	}
}