#include <list>
#include <string>
#include <sstream>
#include "ocr_utils.hpp"
#include "ImageProcessor.h"
#include "ImagesLoader.h"
#include "TextImageProcessor.hpp"

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
 	Mat m = ImagesLoader::openImage(folder, fileName);
	return getInput(m);
}

nn::NeuralFeed ocr::getInput(cv::Mat &img)
{
	ImageProcessor processor(img);

	std::vector<double> hdc = std::move(processor.getHorizontalDensityCurve(HORIZONTAL_DENSITY_POINTS));
	std::vector<double> vdc = std::move(processor.getVerticalDensityCurve(VERTICAL_DENSITY_POINTS));
	std::pair<double, double> c = std::move(processor.getCentroid());
	std::pair<double, double> cc = std::move(processor.getContoursCentroid());
	double nbCountours = processor.getNbContours(NB_MAX_CONTOURS);

	NeuralFeed sent;

	sent.insert(sent.end(), hdc.begin(), hdc.end());
	sent.insert(sent.end(), vdc.begin(), vdc.end());
	sent.push_back(c.first);
	sent.push_back(c.second);
	sent.push_back(cc.first);
	sent.push_back(cc.second);
	sent.push_back(nbCountours);
	return sent;
}

nn::NeuralFeed ocr::getOutput(const nn::NeuralNetwork& network, cv::Mat& img)
{
	return network.update(getInput(img));
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

int ocr::computeImage(const nn::NeuralNetwork &nn, const string &imgFile, ostream &os)
{
	TextImageProcessor processor(nn);
	Mat img = imread(imgFile);
	if (img.data == NULL)
		return -1;

	processor.displayBoundedImage = false;
	processor.debugDisplay = false;
	processor.startProcessing(img);
	
	//imshow("Original Img", img);
	//waitKey(0);
	//destroyWindow("Original Img");


	return 0;
}
