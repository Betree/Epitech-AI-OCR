//
// Created by piouffb on 10/26/15.
//

#include <opencv2/opencv.hpp>
#include "ImageProcessor.h"

ImageProcessor::ImageProcessor(void) {

}

void ImageProcessor::clean(Mat& image) {
    // Concert to greyscale
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // Cropping
    CroppingProfile profile;
    getCroppingProfile(image, profile);
    std::cout << "[Crop] Left:" << profile.left << "px Top:" << profile.top << "px Right:" << profile.right << "px Bottom:" << profile.bottom << "px" << std::endl;
    cv::Rect myRect(profile.left, profile.top, image.cols - profile.right - profile.left, image.rows - profile.top - profile.bottom);
    image = image(myRect);
}

int ImageProcessor::getCroppingHorizontal(const Mat& image, int xStart, int yStart, int xEnd, int yEnd) const
{
    int xIncr = xStart < xEnd ? 1 : -1;
    int yIncr = yStart < yEnd ? 1 : -1;

    for (int x = xStart; x != xEnd; x += xIncr)
    {
        for (int y = yStart; y != yEnd; y += yIncr) {
            if (image.at<unsigned char>(y, x) != 255) {
                return abs(xStart - x);
            }
        }
    }
    //TODO: throw exception : blank image
    return 0;
}

int ImageProcessor::getCroppingVertical(const Mat& image, int xStart, int yStart, int xEnd, int yEnd) const
{
    int xIncr = xStart < xEnd ? 1 : -1;
    int yIncr = yStart < yEnd ? 1 : -1;

    for (int y = yStart; y != yEnd; y += yIncr) {
        for (int x = xStart; x != xEnd; x += xIncr)
        {
            if (image.at<unsigned char>(y, x) != 255) {
                return abs(yStart - y);
            }
        }
    }
    //TODO: throw exception : blank image
    return 0;
}

void ImageProcessor::getCroppingProfile(const Mat &image, ImageProcessor::CroppingProfile &crop) const
{
    bool mustBreak;

    crop.left = getCroppingHorizontal(image, 0, 0, image.cols, image.rows);
    crop.right = getCroppingHorizontal(image, image.cols - 1, 0, -1, image.rows);
    crop.top = getCroppingVertical(image, 0, 0, image.cols, image.rows);
    crop.bottom = getCroppingVertical(image, 0, image.rows - 1, image.cols, -1);
}
