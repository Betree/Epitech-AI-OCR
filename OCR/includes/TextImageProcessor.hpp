#ifndef OCR_TEXTIMAGEPROCESSOR_HPP
#define OCR_TEXTIMAGEPROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <cmath>
#include <string.h>
#include <algorithm>
#include <cstdlib>
#include <NeuralNetwork.hpp>
#include <ocr_utils.hpp>

using namespace cv;
using namespace std;

#define HORIZONTAL	(0)
#define VERTICAL	(1)

class TextImageProcessor
{
private:
	nn::NeuralNetwork _netowrk;
	string				_networkFileName;

	string _fileName;
	bool _drawCreatedHisto;

public:

	// set to true to display debug output
	bool debugDisplay;
	// set to true to display the image with bounded words
	bool displayBoundedImage;

	TextImageProcessor();
	TextImageProcessor(string const fileName);

	string const getFileName() const;
	void setFileName(string const);

	void startProcessing();

private:
	void createBinaryNegativeImage(Mat *, Mat *) const;
	Mat createSmoothedHistogram(Mat, int, int = 7, bool = false) const;
	int getNextMinima(const Mat, unsigned int, int, unsigned int = 0) const;
	int getNextOverMinima(const Mat, unsigned int, int, unsigned int = 0) const;
	vector<Point> getBoundingOfLines(const Mat, const Mat) const;
	vector<Point> getBoundingOfWords(const Mat, const Mat) const;
	vector<Mat> detectWords(Mat);
	vector<pair<Mat, Point> > detectLines(Mat);
};

#endif