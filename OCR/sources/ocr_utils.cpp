#include <list>
#include <sstream>
#include "ocr_utils.hpp"
#include "ImageProcessor.h"
#include "ImagesLoader.h"

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
	Mat img = ImagesLoader::openImage(folder, fileName);
	ImageProcessor processor;

	processor.clean(img);
	std::vector<double> hdc = std::move(processor.getHorizontalDensityCurve(img, HORIZONTAL_DENSITY_POINTS));
	std::vector<double> vdc = std::move(processor.getVerticalDensityCurve(img, VERTICAL_DENSITY_POINTS));

	NeuralFeed sent;

	sent.insert(sent.end(), hdc.begin(), hdc.end());
	sent.insert(sent.end(), vdc.begin(), vdc.end());
	return sent;
}

char ocr::getCharFromOutput(const NeuralFeed& output)
{
	unsigned int idxMax = 0;
	double value = 0;
	for (size_t i = 0; i < output.size(); i++)
	{
		if (output[i] > value)
		{
			idxMax = i;
			value = output[i];
		}
	}
	return (char)idxMax;
}

NeuralFeed ocr::getExpectedOutput(const std::string& fileName)	
{
	char expected = getExpectedChar(fileName);

	if (expected == 0)
		return NeuralFeed();

	NeuralFeed sent(256);
	sent[expected] = 1;
	return sent;
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
