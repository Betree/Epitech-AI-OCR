//
// Created by piouffb on 10/26/15.
//

#ifndef OCR_IMAGESLOADER_H
#define OCR_IMAGESLOADER_H

using namespace cv;

/**
 * Load all .bmp files from given directory
 */
class ImagesLoader {
public:
    ImagesLoader(const std::string &directoryPath);

    ~ImagesLoader();

    bool getNextImage(Mat& image);

private:
    std::vector<std::string> _images;
};


#endif //OCR_IMAGESLOADER_H
