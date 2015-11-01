//
// Created by piouffb on 10/26/15.
//

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "ImagesLoader.h"

static inline char separator()
{
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

ImagesLoader::ImagesLoader(const std::string& directoryPath) {
    DIR* dir;
    struct dirent* dirEnt;
    if ((dir = opendir(directoryPath.c_str())) != NULL) {
        while ((dirEnt = readdir (dir)) != NULL) {
            std::string imagePath(directoryPath);
            imagePath += separator();
            imagePath +=  dirEnt->d_name;
            _images.push_back(imagePath);
        }
        closedir (dir);
    } else {
        perror ("");
        std::cerr << directoryPath << std::endl;
        throw std::runtime_error("Can't load images directory");
    }
}



ImagesLoader::~ImagesLoader() {
}

Mat ImagesLoader::openImage(const std::string & directory, const std::string & filename)
{
	return imread(directory + separator() + filename);
}

bool ImagesLoader::getNextImage(Mat& image) {
    if (_images.size())
    {
        std::string filename = _images.back();
        _images.pop_back();
        std::cout << "Treat image " << filename << std::endl;
        image = imread(filename, 1);
        return true;
    }
    return false;
}
