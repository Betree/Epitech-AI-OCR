#include <iostream>
#include <cmath>
#include "Trainer.hpp"

using namespace nn;
using namespace std;

int main()
{
	NeuralNetwork network(4, 4, std::vector<unsigned int> { 8, 16, 32, 16, 8 });
	NeuralNetwork clone;

	if (!network.save("network.nn"))
		cout << "Failed saving" << endl;
	else if (!clone.load("network.nn"))
		cout << "Failed loading" << endl;
	else if (network.size() != clone.size())
		cout << "network size mismatch";
	else
		for (size_t i = 0; i < network.size(); i++)
		{
			const NeuronLayer& layer(network[i]);
			const NeuronLayer& cloneLayer(clone[i]);

			if (layer.size() != cloneLayer.size())
				cout << i << ": layer size mismatch" << endl;
			else
				for (size_t j = 0; j < layer.size(); j++)
				{
					const Neuron& neuron(layer[j]);
					const Neuron& neuronClone(cloneLayer[j]);

					if (neuron.size() != neuronClone.size())
						cout << i << ", " << j << ": neuron size mismatch" << endl;
					else if (neuron.getBias() != neuronClone.getBias())
						cout << i << ", " << j << ": neuron bias mismatch " << abs(neuron.getBias() - neuronClone.getBias()) << endl;
					else
						for (size_t k = 0; k < neuron.size(); k++)
							if (neuron.getWeight(k) != neuronClone.getWeight(k))
								cout << i << ", " << j << ", " << k << ": weight size mismatch" << endl;
				}
		}
	cin.get();

	//NeuralNetwork network(4, 2, std::vector<unsigned int> { 8 });
	//Trainer::InputOutputPair test;

	//test.first = std::vector<double>{ 0, 1, 0, 1 };
	////test.second = network.update(test.first);
	//test.second = std::vector<double>{ 1, 0.5 };

	//Trainer t(&network);

	//std::string line;

	//do
	//{
	//	t.feed(test);
	//	t.flush();

	//	std::vector<double> output = network.update(test.first);
	//	std::cout << "Error: " << std::endl;
	//	for (size_t i = 0; i < output.size(); i++)
	//	{
	//		std::cout << std::abs(output[i] - test.second[i]) << std::endl;
	//	}
	//	std::cout << "Keep on?" << std::endl;
	//} while (std::getline(std::cin, line));

	return 0;
}