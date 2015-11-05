#include <list>
#include <string>
#include <sstream>
#include "ocr_utils.hpp"
#include "ImageProcessor.h"
#include "ImagesLoader.h"

using namespace std;
using namespace nn;

nn::NeuralNetwork ocr::fromArgv(const std::vector<std::string>& args)
{
	vector<unsigned int> layers;

	for (size_t i = 0; i < args.size(); i++)
	{
		istringstream input(args[i]);
		int layer;

		if (input >> layer)
			layers.push_back(layer);
	}
	return NeuralNetwork(OCR_INPUT_NUMBER, OCR_OUTPUT_NUMBER, layers);
}

NeuralNetwork ocr::fromArgv(unsigned int ac, char** argv)
{
	vector<string> args(ac);

	for (size_t i = 0; i < ac; i++)
	{
		args[i] = argv[i];
	}
	return fromArgv(args);
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

	NeuralFeed sent(OCR_OUTPUT_NUMBER);
	sent[expected] = 1;
	return sent;
}

char ocr::getExpectedChar(const std::string& fileName)
{
	int value = 0;
	istringstream stream(fileName);

	if (!(stream >> value))
		value = fileName[0];
	return value;
}

char ocr::getChar(const NeuralNetwork& network, const string& directory, const string& filename)
{
	return ocr::getCharFromOutput(network.update(getInput(directory, filename)));
}

bool ocr::ocr_test(const NeuralNetwork& network, const std::string& dir, const std::string& file)
{
	NeuralFeed input(std::move(ocr::getInput(dir, file)));
	NeuralFeed output(std::move(network.update(input)));

	char guess = ocr::getCharFromOutput(output);
	char expected = ocr::getExpectedChar(file);

	cout << "Testing " << file << ": Got " << guess << " (" << output[guess] * 100.0 << "%)";
	if (guess != expected)
	{
		cout << " expected " << expected << " (" << output[expected] * 100.0 << "%)";
	}
	cout << endl;
	return guess == expected;
}
