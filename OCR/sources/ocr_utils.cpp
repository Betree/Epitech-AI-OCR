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

NeuralFeed ocr::getInput(const std::string& folder, const std::string& fileName)
{
	// Debug
	// return ocr::getExpectedOutput(fileName);

	throw "Not implemented";
}

char ocr::getCharFromOutput(const NeuralFeed& output)
{
	// Debug
	//unsigned int idxMax = 0;
	//double value = 0;
	//for (size_t i = 0; i < output.size(); i++)
	//{
	//	if (output[i] > value)
	//	{
	//		idxMax = i;
	//		value = output[i];
	//	}
	//}
	//return (char)idxMax;

	throw "Not implemented";
}

NeuralFeed ocr::getExpectedOutput(const std::string& fileName)	
{
	char expected = getExpectedChar(fileName);

	if (expected == 0)
		return NeuralFeed();

	// Debug
	//NeuralFeed sent(256);
	//sent[expected] = 1;
	//return sent;

	throw "Not implemented";
}

char ocr::getExpectedChar(const std::string& fileName)
{
	int value = 0;
	istringstream stream(fileName);

	stream >> value;
	return value;
}

char ocr::getChar(const NeuralNetwork& network, const string& directory, const string& filename)
{
	return ocr::getCharFromOutput(network.update(getInput(directory, filename)));
}
