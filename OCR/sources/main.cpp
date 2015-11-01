#include <opencv2/opencv.hpp>
#include "ImagesLoader.h"
#include "ImageProcessor.h"

using namespace nn;
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	ImagesLoader loader("/home/piouffb/workspace/EPITECH/Epitech-AI-OCR/OCR/data/55x_small_dataset/55x_small_dataset");
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
