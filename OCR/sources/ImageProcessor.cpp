//
// Created by piouffb on 10/26/15.
//

#include <opencv2/opencv.hpp>
#include "ImageProcessor.h"

ImageProcessor::ImageProcessor(void) {

}

void ImageProcessor::clean(Mat& image) {
    CroppingProfile profile;
    getCroppingProfile(image, profile);
    std::cout << "[Crop] Left:" << profile.left << "px Top:" << profile.top << "px Right:" << profile.right << "px Bottom:" << profile.bottom << "px" << std::endl;
    cv::Rect myRect(profile.left, profile.top, image.size().width - profile.right - profile.left, image.size().height - profile.top - profile.bottom);
    image = image(myRect);
}

void ImageProcessor::getCroppingProfile(const Mat &image, ImageProcessor::CroppingProfile &crop) const
{
    bool mustBreak;

    // Croping left
    mustBreak = false;
    for (int x = 0; x < image.size().width; ++x)
    {
        for (int y = 0; y < image.size().height; ++y) {
            if (isWhite(image, image.at<Vec3b>(x, y))) {
                mustBreak = true;
                break;
            }
        }
        if (mustBreak) {
            crop.left = x;
            break;
        }
    }

    // Croping right
    mustBreak = false;
    for (int x = image.size().width; x > 0; --x)
    {
        for (int y = 0; y < image.size().height; ++y) {
            if (isWhite(image, image.at<Vec3b>(x, y))) {
                mustBreak = true;
                break;
            }
        }
        if (mustBreak) {
            crop.right = image.size().width - x;
            break;
        }
    }

    // Croping top
    mustBreak = false;
    for (int y = 0; y < image.size().height; ++y)
    {
        for (int x = 0; x < image.size().width; ++x) {
            if (isWhite(image, image.at<Vec3b>(x, y))) {
                mustBreak = true;
                break;
            }
        }
        if (mustBreak) {
            crop.top = y;
            break;
        }
    }

    // Croping bottom
    mustBreak = false;
    for (int y = image.size().height; y > 0; --y)
    {
        for (int x = 0; x < image.size().width; ++x) {
            if (isWhite(image, image.at<Vec3b>(x, y))) {
                mustBreak = true;
                break;
            }
        }
        if (mustBreak) {
            crop.bottom = image.size().height - y;
            break;
        }
    }
}

bool ImageProcessor::isWhite(const Mat& image, const Vec3b &pixel) const {
    for(int k = 0; k < image.channels(); k++) {
        uchar col = pixel.val[k];
        if (col == 255)
            return false;
    }
    return true;
}
