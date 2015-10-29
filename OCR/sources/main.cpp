#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "ImagesLoader.h"
#include "ImageProcessor.h"

using namespace cv;

int main(int argc, char** argv)
{
/*	if (argc != 2)
	{
		printf("usage: OCR <Image_Path>\n");
		return -1;
	}

	Mat image;
	image = imread(argv[1], 1);
	if (!image.data)
	{
		printf("No image data \n");
		return -1;
	}

	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", image);
	waitKey(0);
*/
	ImagesLoader loader("/home/piouffb/workspace/EPITECH/Epitech-AI-OCR/OCR/data/55x_small_dataset/55x_small_dataset");
	ImageProcessor processor;
	Mat image;
	while (loader.getNextImage(image))
	{
		// Clean image
		processor.clean(image);

		// Temp : Aff image
		namedWindow("Display Image", WINDOW_AUTOSIZE);
		imshow("Display Image", image);
		waitKey(0);
		return 0;
	}
	return 0;
}
