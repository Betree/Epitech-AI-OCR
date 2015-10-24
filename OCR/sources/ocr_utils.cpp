#include <list>
#include <sstream>
#include "ocr_utils.hpp"

using namespace std;
using namespace nn;

NeuralNetwork ocr::fromArgv(unsigned int ac, char** argv)
{
	vector<unsigned int> layers;

	for (size_t i = 0; i < ac; i++)
	{
		istringstream input(argv[i]);
		int layer;

		if (input >> layer)
			layers.push_back(layer);
		--ac;
	}
	return NeuralNetwork(OCR_INPUT_NUMBER, OCR_OUTPUT_NUMBER, layers);
}

NeuralFeed ocr::getInput(const std::string& fileName)
{
	throw "Not implemented";
}

char ocr::getCharFromOutput(const NeuralFeed& output)
{
	throw "Not implemented";
}

NeuralFeed ocr::getExpectedOutput(const std::string& fileName)	
{
	char expected = getExpectedChar(fileName);

	if (expected == 0)
		return NeuralFeed();
	throw "Not implemented";
}

char ocr::getExpectedChar(const std::string& fileName)
{
	int value = 0;
	istringstream stream(fileName);

	stream >> value;
	return value;
}

char ocr::getChar(const NeuralNetwork& network, const string& filename)
{
	return ocr::getCharFromOutput(network.update(getInput(filename)));
}
