#define CTOR_DECLARE(name)		name();
#define DTOR_DECLARE(name)		~name();
#define CCTOR_DECLARE(name)	name(const name& other);
#define COPY_DECLARE(name)		name&	operator=(const name& other);

#define COPLIEN_DECLARE(name)	CTOR_DECLARE(name)	CCTOR_DECLARE(name)	DTOR_DECLARE(name)	COPY_DECLARE(name)

#include <vector>
#include <string>

namespace nn
{
	class Neuron
	{
		public:
			Neuron(int inputNumber); // TODO init random
			~Neuron();
			Neuron(const Neuron& other);
			Neuron(const Neuron&& other);
			Neuron&	operator=(const Neuron& other);

			double& operator[](unsigned int idx);
			double operator[](unsigned int idx) const;
			
		private:
			std::vector<double> _weights;
	};
	
	class NeuronLayer
	{
		public:
			NeuronLayer(unsigned int inputNumber, unsigned int );
			~NeuronLayer();
			NeuronLayer(const NeuronLayer& other);
			NeuronLayer(const NeuronLayer&& other);
			NeuronLayer& operator=(const NeuronLayer& other);

			Neuron& operator[](unsigned int idx);
			const Neuron& operator[](unsigned int idx) const;
			
		private:
			std::vector<Neuron*> _neurons;
	};
	
	class NeuralNetwork
	{
		public:
			NeuralNetwork();
			~NeuralNetwork();
			NeuralNetwork(const NeuralNetwork& other);
			NeuralNetwork(const NeuralNetwork&& other);
			NeuralNetwork& operator=(const NeuralNetwork& other);
			
			std::vector<std::pair<bool, float> > update(const std::vector<bool>& input) const;
			
			void save(const std::string& filename) const;
			void load(const std::string& filename);
			
		private:
			std::vector<NeuronLayer*> _layers;
	};
}
