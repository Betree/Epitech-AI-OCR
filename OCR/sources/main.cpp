#include <opencv2/opencv.hpp>
#include "NeuralNetwork.hpp"
#include "ocr_utils.hpp"

using namespace nn;
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if (argc != 4)
	{
		cout << "Usage: " << argv[0] << " folder file network_file" << endl;
		return 1;
	}
	
	NeuralNetwork network;

	if (!network.load(argv[3]))
	{
		cerr << "Unable to load network file: " << argv[3] << endl;
		return 1;
	}

	NeuralFeed input(std::move(ocr::getInput(argv[1], argv[2])));
	NeuralFeed output(std::move(network.update(input)));

	char guess = ocr::getCharFromOutput(output);
	cout << "I think this is a: " << guess << " (" << (int)guess << ") with " << output[guess] * 100.0 << '%' << endl;
	char expected = ocr::getExpectedChar(argv[2]);
	cout << "Expected: " << expected << " (" << (int)expected << ") with " << output[expected] * 100.0 << '%' << endl;
	return 0;
}
