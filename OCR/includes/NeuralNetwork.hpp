/*
#define CTOR_DECLARE(name)		name();
#define DTOR_DECLARE(name)		~name();
#define CCTOR_DECLARE(name)	name(const name& other);
#define COPY_DECLARE(name)		name&	operator=(const name& other);

#define COPLIEN_DECLARE(name)	CTOR_DECLARE(name)	CCTOR_DECLARE(name)	DTOR_DECLARE(name)	COPY_DECLARE(name)
*/

#ifndef	NEURALNETWORK_HPP_
#define	NEURALNETWORK_HPP_

#include <vector>
#include <string>

namespace nn
{
	class Neuron
	{
		public:
			Neuron(unsigned int inputNumber); // TODO init random
			~Neuron();
			Neuron(const Neuron& other);
			Neuron(const Neuron&& other);
			Neuron&	operator=(const Neuron& other);

			double& getWeight(unsigned int idx);
			double getWeight(unsigned int idx) const;
			void setWeight(unsigned int idx, double value);
			size_t size() const;
			
			double getBias() const;
			void setBias(double value);
			
		private:
			std::vector<double>	_weights;
			double _bias;
	};
	
	class NeuronLayer
	{
		public:
			NeuronLayer(unsigned int inputNumber, unsigned int neuronNumber);
			~NeuronLayer();
			NeuronLayer(const NeuronLayer& other);
			NeuronLayer(const NeuronLayer&& other);
			NeuronLayer& operator=(const NeuronLayer& other);

			Neuron& operator[](unsigned int idx);
			const Neuron& operator[](unsigned int idx) const;
			unsigned int size() const;
			unsigned int getInputNumber() const;
			
		private:
			std::vector<Neuron*> _neurons;
			unsigned _inputNumber;
	};

	typedef std::vector<double>	NeuralFeed;

	class NeuralNetwork
	{
		public:
			NeuralNetwork(unsigned int inputNumber, unsigned int outputNumber, const std::vector<unsigned int>& hiddenLayersDefinition);
			NeuralNetwork();
			~NeuralNetwork();
			NeuralNetwork(const NeuralNetwork& other);
			NeuralNetwork(const NeuralNetwork&& other);
			NeuralNetwork& operator=(const NeuralNetwork& other);

			NeuronLayer& operator[](unsigned int idx);
			const NeuronLayer& operator[](unsigned int idx) const;
			unsigned int size() const;

			static double sigmoid_prime(double value);
			static double sigmoid(double value);

			NeuralFeed update(NeuralFeed input) const;
			
			bool save(const std::string& filename) const;
			bool load(const std::string& filename);
			
		private:
			void cleanup();

			std::vector<NeuronLayer*> _layers;
	};
}

#endif	// NEURALNETWORK_HPP_