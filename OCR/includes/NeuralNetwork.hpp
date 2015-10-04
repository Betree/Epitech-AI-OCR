/*
#define CTOR_DECLARE(name)		name();
#define DTOR_DECLARE(name)		~name();
#define CCTOR_DECLARE(name)	name(const name& other);
#define COPY_DECLARE(name)		name&	operator=(const name& other);

#define COPLIEN_DECLARE(name)	CTOR_DECLARE(name)	CCTOR_DECLARE(name)	DTOR_DECLARE(name)	COPY_DECLARE(name)
*/

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
			
			double getThreshold() const;
			void setThreshold(double value);
			
		private:
			std::vector<double>	_weights;
			double						_threshold;
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
			unsigned int				_inputNumber;
	};
	
	class NeuralNetwork
	{
		public:
			NeuralNetwork(unsigned int inputNumber, unsigned int outputNumber, const std::vector<unsigned int>& hiddenLayersDefinition);
			~NeuralNetwork();
			NeuralNetwork(const NeuralNetwork& other);
			NeuralNetwork(const NeuralNetwork&& other);
			NeuralNetwork& operator=(const NeuralNetwork& other);
			
			std::vector<std::pair<bool, double> > update(const std::vector<bool>& input) const;
			
			void save(const std::string& filename) const;
			void load(const std::string& filename);
			
		private:
			std::vector<NeuronLayer*> _layers;
	};
}
