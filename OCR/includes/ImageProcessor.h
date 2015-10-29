//
// Created by piouffb on 10/26/15.
//

#ifndef OCR_IMAGEPROCESSOR_H
#define OCR_IMAGEPROCESSOR_H

using namespace cv;

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
     * [!] Method behavior is undefined in case nbPoints is greater than image height
     *
     * returns a vector of nbPoints size filled with densities represented with double between 0.0 and 1.0
     */
    std::vector<double> getHorizontalDensityCurve(Mat& image, int nbPoints = 10) const;

    /*
     * Get the horizontal density curve (the density for each columns)
     * [!] Method behavior is undefined in case nbPoints is greater than image width
     *
     * returns a vector of nbPoints size filled with densities represented with double between 0.0 and 1.0
     */
    std::vector<double> getVerticalDensityCurve(Mat& image, int nbPoints = 10) const;

private:
    /*
     * Cropping
     */
    void getCroppingProfile(const Mat& image, CroppingProfile& profile) const;
    int getCroppingHorizontal(const Mat& image, int xStart, int yStart, int xEnd, int yEnd) const;
    int getCroppingVertical(const Mat& image, int xStart, int yStart, int xEnd, int yEnd) const;
};


#endif //OCR_IMAGEPROCESSOR_H
