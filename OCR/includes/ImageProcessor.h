//
// Created by piouffb on 10/26/15.
//

#ifndef OCR_IMAGEPROCESSOR_H
#define OCR_IMAGEPROCESSOR_H

#include <opencv2/opencv.hpp>

using namespace cv;

#define BLACK_THRESHOLD 100

class ImageProcessor {
public:
    struct CroppingProfile
    {
        int top; int bottom; int left; int right;
    };

public:
    ImageProcessor(void);
    void clean(Mat& image);

/*
 * Features extractors
 */
    /*
     * Get the horizontal density curve (the density for each rows)
     * Curve is filled with 0 density on each side to match nbPoints size if the image is too small
     *
     * returns a vector of nbPoints size filled with densities represented with double between 0.0 and 1.0
     */
    std::vector<double> getHorizontalDensityCurve(const Mat& image, int nbPoints = 10) const;

    /*
     * Get the horizontal density curve (the density for each columns)
     * Curve is filled with 0 density on each side to match nbPoints size if the image is too small
     *
     * returns a vector of nbPoints size filled with densities represented with double between 0.0 and 1.0
     */
    std::vector<double> getVerticalDensityCurve(const Mat& image, int nbPoints = 10) const;

    /*
     * Returns image centroid
     */
    std::pair<double, double> getCentroid(const Mat& image) const;

    /*
     * Returns the centroid generated with figure's contours
     */
    std::pair<double, double> getContoursCentroid(const Mat &image) const;

private:
    /*
     * Cropping
     */
    void getCroppingProfile(const Mat& image, CroppingProfile& profile) const;
    int getCroppingHorizontal(const Mat& image, int xStart, int yStart, int xEnd, int yEnd) const;
    int getCroppingVertical(const Mat& image, int xStart, int yStart, int xEnd, int yEnd) const;

    /*
     * Density curves
     */
    void completeCurveWithBlank(std::vector<double>& curve, int nbPoints) const;
};


#endif //OCR_IMAGEPROCESSOR_H
