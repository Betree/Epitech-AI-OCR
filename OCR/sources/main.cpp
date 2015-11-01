#include <opencv2/opencv.hpp>
#include "NeuralNetwork.hpp"
#include "ImagesLoader.h"
#include "ImageProcessor.h"

using namespace nn;
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "Usage: " << argv[0] << " dataset_folder" << endl;
		return 1;
	}
	
	ImagesLoader loader(argv[1]);
	ImageProcessor processor;
	Mat image;
	while (loader.getNextImage(image))
	{
		// Clean image
		processor.clean(image);
		processor.getHorizontalDensityCurve(image, 8);
		processor.getVerticalDensityCurve(image, 8);

		// Temp : Aff image and returns
		namedWindow("Display Image", WINDOW_AUTOSIZE);
		imshow("Display Image", image);
		waitKey(0);
		return 0;
	}
	return 0;
}
