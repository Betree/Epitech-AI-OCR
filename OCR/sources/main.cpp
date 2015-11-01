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

	cout << "I think this is a: " << ocr::getCharFromOutput(output) << endl;

	cin.get();
	return 0;
}
