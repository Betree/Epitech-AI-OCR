//
// Created by piouffb on 10/26/15.
//

#include <vector>
#include <opencv2/opencv.hpp>
#include "ImageProcessor.h"

using namespace std;

ImageProcessor::ImageProcessor(Mat& image) : _image(image)
{
    // Concert to greyscale if necessary
    if (_image.channels() > 1)
        cv::cvtColor(_image, _image, cv::COLOR_BGR2GRAY);

    // Cropping
    CroppingProfile profile;
    getCroppingProfile(_image, profile);
    cv::Rect myRect(profile.left, profile.top, _image.cols - profile.right - profile.left, _image.rows - profile.top - profile.bottom);
    _image = _image(myRect);
}

int ImageProcessor::getCroppingHorizontal(const Mat& image, int xStart, int yStart, int xEnd, int yEnd) const
{
    int xIncr = xStart < xEnd ? 1 : -1;
    int yIncr = yStart < yEnd ? 1 : -1;

    for (int x = xStart; x != xEnd; x += xIncr) {
        for (int y = yStart; y != yEnd; y += yIncr) {
            if (image.at<unsigned char>(y, x) <= BLACK_THRESHOLD) {
                return abs(xStart - x);
            }
			else if (image.at<unsigned char>(y, x) != 255)
			{
//				cout << (unsigned int)image.at<unsigned char>(y, x) << endl;
			}
        }
    }
    //TODO: throw exception : blank image
 //   std::cerr << "Warning: Image is blank" << std::endl;
    return 0;
}

int ImageProcessor::getCroppingVertical(const Mat& image, int xStart, int yStart, int xEnd, int yEnd) const
{
    int xIncr = xStart < xEnd ? 1 : -1;
    int yIncr = yStart < yEnd ? 1 : -1;

    for (int y = yStart; y != yEnd; y += yIncr) {
        for (int x = xStart; x != xEnd; x += xIncr) {
            if (image.at <unsigned char>(y, x) <= BLACK_THRESHOLD) {
                return abs(yStart - y);
            }
        }
    }
    //TODO: throw exception : blank image
   // std::cerr << "Warning: Image is blank" << std::endl;
    return 0;
}

void ImageProcessor::getCroppingProfile(const Mat& image, ImageProcessor::CroppingProfile &crop) const
{
    crop.left = getCroppingHorizontal(image, 0, 0, image.cols, image.rows);
    crop.right = getCroppingHorizontal(image, image.cols - 1, 0, -1, image.rows);
    crop.top = getCroppingVertical(image, 0, 0, image.cols, image.rows);
    crop.bottom = getCroppingVertical(image, 0, image.rows - 1, image.cols, -1);
}

std::vector<double> ImageProcessor::getHorizontalDensityCurve(unsigned int nbPoints) const
{
    std::vector<double> weightCurve;
    weightCurve.reserve(nbPoints);

    double storeThreshold = (double) _image.rows / (double) nbPoints;
    double currentThreshold = storeThreshold;
    double densityBuffer = 0.0;
    int nbRowsAnalysed = 0;
    int lastRowStored = 0;
    for (int y = 0; y != _image.rows; ++y) {
        int rowTotalPixelsDensity = 0;
        for (int x = 0; x != _image.cols; ++x)
            rowTotalPixelsDensity += 255 - _image.at<unsigned char>(y, x);
        densityBuffer += rowTotalPixelsDensity / (double) _image.cols / 255.0;
        nbRowsAnalysed += 1;

        // If average current lines densities must be stored in vector
        if ((double) nbRowsAnalysed >= currentThreshold) {
            weightCurve.push_back(densityBuffer / (nbRowsAnalysed - lastRowStored));
            densityBuffer = 0.0;
            currentThreshold += storeThreshold;
            lastRowStored = nbRowsAnalysed;
        }
    }

    // Check everything's fine
    if (lastRowStored != nbRowsAnalysed)
        weightCurve.push_back(densityBuffer / (nbRowsAnalysed - lastRowStored));
    else if (weightCurve.size() > nbPoints)
        throw std::runtime_error("Error: TOO MUCH DATA");

    // Complete data with blank, useful when picture is smaller than nbPoints
    if (weightCurve.size() < nbPoints)
        completeCurveWithBlank(weightCurve, nbPoints);
    return weightCurve;
}

std::vector<double> ImageProcessor::getVerticalDensityCurve(unsigned int nbPoints) const
{
    std::vector<double> weightCurve;
    weightCurve.reserve(nbPoints);

    double storeThreshold = (double) _image.cols / (double) nbPoints;
    double currentThreshold = storeThreshold;
    double densityBuffer = 0.0;
    int nbColsAnalysed = 0;
    int lastColStored = 0;
    for (int x = 0; x != _image.cols; ++x) {
        int colTotalPixelsDensity = 0;
        for (int y = 0; y != _image.rows; ++y)
            colTotalPixelsDensity += 255 - _image.at<unsigned char>(y, x);
        densityBuffer += colTotalPixelsDensity / (double) _image.rows / 255.0;
        nbColsAnalysed += 1;

        // If average current lines densities must be stored in vector
        if ((double) nbColsAnalysed >= currentThreshold) {
            weightCurve.push_back(densityBuffer / (nbColsAnalysed - lastColStored));
            densityBuffer = 0.0;
            currentThreshold += storeThreshold;
            lastColStored = nbColsAnalysed;
        }
    }

    // Check everything's fine
    if (lastColStored != nbColsAnalysed)
        weightCurve.push_back(densityBuffer / (nbColsAnalysed - lastColStored));
    else if (weightCurve.size() > nbPoints)
        throw std::runtime_error("Error: too much data");

    // Complete data with blank, useful when picture is smaller than nbPoints
    if (weightCurve.size() < nbPoints)
        completeCurveWithBlank(weightCurve, nbPoints);
    return weightCurve;
}

void ImageProcessor::completeCurveWithBlank(std::vector<double> &curve, unsigned int nbPoints) const
{
    bool appendLeft = false;

    while (curve.size() < nbPoints) {
        if (appendLeft)
            curve.insert(curve.begin(), 0.0);
        else
            curve.push_back(0.0);
        appendLeft = !appendLeft;
    }
}

std::pair<double, double> ImageProcessor::getCentroid() const
{
    // By default centroid is at the center of picture
    Point center(_image.cols / 2, _image.rows / 2);
    int imageSize = _image.cols * _image.rows;
    int nbValues = 0;

    // Calculate centroid
    for (int imgPos = 0; imgPos < imageSize; ++imgPos) {
        int x = imgPos % _image.cols;
        int y = imgPos / _image.cols;
        if (_image.at<unsigned char> (y, x) <= BLACK_THRESHOLD)
        {
            center.x += x;
            center.y += y;
            ++nbValues;
        }
    }

    if (nbValues) {
        center.x /= nbValues;
        center.y /= nbValues;
    }

    return std::pair<double, double>((double) center.x / (double) _image.cols, (double) center.y / _image.rows);
}

std::pair<double, double> ImageProcessor::getContoursCentroid()
{
    initContours();

    // Get the moments
    vector<Moments> mu(_contours.size());
    for (unsigned int i = 0; i < _contours.size(); i++)
        mu[i] = moments(_contours[i], false);

    //  Get the mass center
    Point center(0, 0);
    int nbMasses = 0;
    for (unsigned int i = 0; i < _contours.size(); i++) {
        if (mu[i].m00) {
            center.x += (int)(mu[i].m10 / mu[i].m00);
            center.y += (int)(mu[i].m01 / mu[i].m00);
            ++nbMasses;
        }
    }
    if (nbMasses) {
        center.x /= nbMasses;
        center.y /= nbMasses;
    }
    else // Default if no contour has been found : take the center of image
    {
        center.x = _image.cols / 2;
        center.y = _image.rows / 2;
    }

    return std::pair<double, double>((double) center.x / (double) _image.cols, (double) center.y / _image.rows);
}

double ImageProcessor::getNbContours(unsigned int maxNbContours)
{
    initContours();
    if (_contours.size() >= maxNbContours)
        return 1.0;
    return (double)_contours.size() / (double)maxNbContours;
}

void ImageProcessor::initContours()
{
    if (!_contours.size())
    {
        Mat edges;
        vector<Vec4i> hierarchy;

        // Detect edges
        Canny(_image, edges, BLACK_THRESHOLD, BLACK_THRESHOLD * 2, 3);

        // Find contours
        findContours(edges, _contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    }
}
