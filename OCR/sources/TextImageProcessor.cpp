#include "TextImageProcessor.hpp"


TextImageProcessor::TextImageProcessor()
{
	this->_fileName.clear();
	this->_drawCreatedHisto = false;
	this->debugDisplay = false;
	this->displayBoundedImage = false;
}

TextImageProcessor::TextImageProcessor(string fileName) : _fileName(fileName)
{
	this->_drawCreatedHisto = false;
	this->debugDisplay = false;
	this->displayBoundedImage = false;
}

string const TextImageProcessor::getFileName() const
{
	return this->_fileName;
}

void TextImageProcessor::setFileName(string const fileName)
{
	this->_fileName = fileName;
}

void TextImageProcessor::startProcessing()
{
	if (this->_fileName.empty())
		return;

	Mat binary;
	vector<vector<Mat> > wordInOrder;
	cv::Mat img = cv::imread(this->_fileName);


	createBinaryNegativeImage(&img, &binary);
	vector<pair<Mat, Point> > lines = detectLines(binary);

	for (int i = 0; i < lines.size(); ++i)
	{	
		wordInOrder.push_back(detectWords(lines[i].first));

		if (displayBoundedImage)
		{
			line(binary, Point(0, lines[i].second.x), Point(binary.size().width - 1, lines[i].second.x), Scalar(255));
			line(binary, Point(0, lines[i].second.y), Point(binary.size().width - 1, lines[i].second.y), Scalar(125));
		}
	}


	// Display the new binary image
	if (displayBoundedImage)
	{
		namedWindow("Binary Image", 1);
		imshow("Binary Image", binary);
	}
}

void TextImageProcessor::createBinaryNegativeImage(Mat *img, Mat *output) const
{
	Mat greyImg;
	double Imin, Imax;
	Point minLoc, maxLoc;

	cvtColor(*img, *img, CV_BGR2GRAY);

	// initialize the output matrix with zeros
	Mat negative = Mat::zeros(img->size(), img->type());

	// create a matrix with all elements equal to 255 for subtraction
	Mat sub_mat = Mat::ones(img->size(), img->type()) * 255;

	//subtract the original matrix by sub_mat to give the negative output negative
	subtract(sub_mat, *img, negative);

	// Get the Imax, Imin and Imean
	minMaxLoc(negative, &Imin, &Imax, NULL, NULL);

	// Display them
	if (debugDisplay)
	{
		cout << "minimum intensity : " << Imin << endl;
		cout << "Maximum intensity : " << Imax << endl;
	}

	// calculate Idiff
	double Idiff = Imax - Imin;
	// So we can calculate Izero..
	double Izero = Idiff > (255 / 2) ? 100 : 20;
	// ... and get the treshold value
	double Itreshold = Imin + Izero;

	//create a binary image depending on the Itreshold
	threshold(negative, *output, Itreshold, 255, 3);
}

Mat TextImageProcessor::createSmoothedHistogram(const Mat img, int t, int blurHardness, bool gaussian) const
{
	//col or row histogram?  
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_8U);

	//count nonzero value and check max V  
	int max = -100;
	for (int j = 0; j < sz; ++j)
	{
		Mat data = (t) ? img.row(j) : img.col(j);
		int v = countNonZero(data);
		mhist.at< unsigned char >(j) = v;
		if (v > max)
			max = v;
	}

	// It's used to "smooth" the histogram
	medianBlur(mhist, mhist, blurHardness);

	// When creating word histogram we need to spread them to cover small gab between letters
	if (gaussian)
		GaussianBlur(mhist, mhist, Size(25, 1), 0, 0); // -> pas cool, a changer si on a le temps

	if (this->_drawCreatedHisto)
	{
		Mat histo;
		int width, height;
		if (t)
		{
			width = max;
			height = sz;
			histo = Mat::zeros(Size(width, height), CV_8U);

			for (int i = 0; i < height; ++i)
			{
				for (int j = 0; j < mhist.at< unsigned char >(i); ++j)
					histo.at< unsigned char >(i, j) = 255;
			}
		}
		else{
			width = sz;
			height = max;
			histo = Mat::zeros(Size(width, height), CV_8U);

			for (int i = 0; i < width; ++i)
			{
				for (int j = 0; j < mhist.at< unsigned char >(i); ++j)
					histo.at< unsigned char >(max - j - 1, i) = 255;
			}
		}
		// Display the histogram image
		namedWindow("Histogram", 1);
		imshow("Histogram", histo);
	}
	return mhist;
}

int TextImageProcessor::getNextMinima(const Mat vhist, unsigned int index, int direction, unsigned int treshold) const
{
	int size = direction ? vhist.cols : vhist.rows;
	while (index < size && vhist.at<unsigned char>(index) > treshold)
		++index;
	if (index >= size)
		return -1;
	return index;
}

int TextImageProcessor::getNextOverMinima(const Mat vhist, unsigned int index, int direction, unsigned int treshold) const
{
	int size = direction ? vhist.cols : vhist.rows;
	while (index < size && vhist.at<unsigned char>(index) <= treshold)
		++index;
	if (index >= size)
		return -1;
	return index;
}

vector<Point> TextImageProcessor::getBoundingOfLines(const Mat img, const Mat vhist) const
{
	int index = 0;
	int oldIndex = 0;
	vector<Point> ret;
	while (index != -1)
	{
		index = getNextOverMinima(vhist, index, VERTICAL, 10);
		if (index == -1) break;
		if (debugDisplay)
			cout << "Debut de la ligne : " << index << ", ";
		oldIndex = index;
		index = getNextMinima(vhist, index, VERTICAL, 10);
		if (debugDisplay)
			cout << " fin de la ligne : " << index << endl;

		if (index - oldIndex > img.size().height / 80)
		{
			ret.push_back(Point(oldIndex, index));
		}
	}
	return ret;
}

vector<Point> TextImageProcessor::getBoundingOfWords(const Mat img, const Mat hist) const
{
	int index = 0;
	int oldIndex = 0;
	vector<Point> ret;
	while (index != -1)
	{
		index = getNextOverMinima(hist, index, VERTICAL, 0);
		if (index == -1) break;
		if (debugDisplay)
			cout << "Debut du mot : " << index << ", ";
		oldIndex = index;
		index = getNextMinima(hist, index, VERTICAL, 0);
		if (debugDisplay)
			cout << " fin du mot : " << index << endl;
		ret.push_back(Point(oldIndex, index));
	}
	return ret;
}

vector<Mat> TextImageProcessor::detectWords(Mat lines)
{
	Mat extractedLineHHist;
	vector<Point> wordBounding;
	vector<Mat> ret;

	extractedLineHHist = createSmoothedHistogram(lines, HORIZONTAL, 3, true);
	wordBounding = getBoundingOfWords(lines, extractedLineHHist);

	Mat extractedWord;
	for (int i = 0; i < wordBounding.size(); ++i)
	{
		//get sub image of lines out of the main picture
		extractedWord = lines(Rect(wordBounding[i].x, 0, wordBounding[i].y - wordBounding[i].x, lines.rows));
		ret.push_back(extractedWord);
		// Maintenant on detecte les mots pour chaque ligne
		//
		if (displayBoundedImage)
		{
			line(lines, Point(wordBounding[i].x, 0), Point(wordBounding[i].x, lines.rows - 1), Scalar(255));
			line(lines, Point(wordBounding[i].y, 0), Point(wordBounding[i].y, lines.rows - 1), Scalar(125));
		}
	}
	return ret;
}

vector<pair<Mat, Point> > TextImageProcessor::detectLines(cv::Mat binary)
{
	Mat vhist = createSmoothedHistogram(binary, VERTICAL);

	vector<Point> boundingLines = getBoundingOfLines(binary, vhist);
	vector<pair<Mat, Point> > ret;

	Mat extractedLine;
	for (int i = 0; i < boundingLines.size(); ++i)
	{
		//get sub image of lines out of the main picture
		extractedLine = binary(Rect(0, boundingLines[i].x, binary.cols, boundingLines[i].y - boundingLines[i].x));
		ret.push_back(pair<Mat, Point>(extractedLine, boundingLines[i]));
	}

	return ret;
}
