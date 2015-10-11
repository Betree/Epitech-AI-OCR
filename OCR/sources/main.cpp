#include <iostream>
#include <cmath>
#include "Trainer.hpp"

using namespace nn;

int main()
{
	NeuralNetwork network(4, 2, std::vector<unsigned int> { 8 });
	Trainer::InputOutputPair test;

	test.first = std::vector<double>{ 0, 1, 0, 1 };
	//test.second = network.update(test.first);
	test.second = std::vector<double>{ 1, 0.5 };

	Trainer t(&network);

	std::string line;

	do
	{
		t.feed(test);
		t.flush();

		std::vector<double> output = network.update(test.first);
		std::cout << "Error: " << std::endl;
		for (size_t i = 0; i < output.size(); i++)
		{
			std::cout << std::abs(output[i] - test.second[i]) << std::endl;
		}
		std::cout << "Keep on?" << std::endl;
	} while (std::getline(std::cin, line));

	return 0;
}