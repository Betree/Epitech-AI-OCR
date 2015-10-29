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
private:
    void getCroppingProfile(const Mat& image, CroppingProfile& profile) const;
    int getCroppingHorizontal(const Mat& image, int xStart, int yStart, int xEnd, int yEnd) const;
    int getCroppingVertical(const Mat& image, int xStart, int yStart, int xEnd, int yEnd) const;
};


#endif //OCR_IMAGEPROCESSOR_H
