#include <sstream>
#include <iostream>
#include <cmath>
#include <string.h>
#include <algorithm>

#ifdef WIN32
# include <direct.h>
# include "dirent.h"
# define chdir _chdir
# define strerror _strerror
#else
# include <dirent.h>
#include <ImageProcessor.h>
#include <ImagesLoader.h>

#endif

#include "ocr_utils.hpp"
#include "Trainer.hpp"

using namespace nn;
using namespace std;

char* _strerror(int err)
{
	static char buffer[256];

	#ifdef WIN32
		strerror_s(buffer, err);
    #else
		strerror_r(err, buffer, 256);
    #endif
	return buffer;
}

int print_usage(const char* exeName)
{
	cout << "Usage: " << exeName << " dataset_folder minibatch_size network_file [network_layer_size...]" << endl;
	return 1;
}

double distance(const NeuralNetwork& network, const Trainer::Epoch& epoch)
{
	double distance = 0;
	size_t inputCount = 0;

	for (; inputCount < epoch.size(); ++inputCount)
	{
		NeuralFeed output(std::move(network.update(epoch[inputCount].first)));
		const NeuralFeed& expected(epoch[inputCount].second);

		double tmp = 0;
		for (size_t i = 0; i < epoch[i].second.size(); i++)
			tmp += abs(expected[i] - output[i]);
		distance += tmp / expected.size();
	}

	if (inputCount == 0)
		++inputCount;
	return distance / inputCount;
}

int ocr_training(const string& dataset_folder, unsigned int minibatch_size, const string& network_file, int ac, char** av)
{
	NeuralNetwork network;
	Trainer trainer(&network);

	trainer.setMiniBatchSize(minibatch_size);
	/*if (ac > 0)
		network = ocr::fromArgv(ac, av);
	else if (!network.load(network_file))
	{
		cerr << "Unable to load network from " << network_file << endl;
		return 2;
	}*/


	DIR* dir;
	if (!(dir = opendir(dataset_folder.c_str())))
	{
		cerr << "Unable to open " << dataset_folder << ": " << strerror(errno) << endl;
		return 3;
	}

	Trainer::Epoch epoch;
	struct dirent* ent;

	while ((ent = readdir(dir)))
	{
		string filename(ent->d_name);
		ImageProcessor processor;
		ImagesLoader loader(dataset_folder);
		if (filename.length() >= 4 && !filename.compare(filename.length() - 4, 4, ".bmp"))
		{
			/*NeuralFeed output(std::move(ocr::getExpectedOutput(filename)));

			if (!output.empty()) {
				NeuralFeed input(std::move(ocr::getInput(dataset_folder, filename)));

				epoch.push_back(Trainer::InputOutputPair(input, output));
			}*/
			Mat image = loader.openImage(dataset_folder, filename);
			processor.clean(image);
			std::pair<double, double> centroid = processor.getCentroid(image);
			std::pair<double, double> contoursCentroid = processor.getContoursCentroid(image);

			cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
			circle(image, Point(centroid.first * image.cols, centroid.second * image.rows), 2, Scalar(0, 255, 100), -1, 8, 0);
			circle(image, Point(contoursCentroid.first * image.cols, contoursCentroid.second * image.rows), 2, Scalar(255, 0, 0), -1, 8, 0);

			/// Show in a window
			namedWindow("Contours", WINDOW_NORMAL);
			imshow("Contours", image);
			waitKey(0);
		}
	}
	return 0;
	/*closedir(dir);

	cout << "[DEBUG] Distance before: " << distance(network, epoch) << endl;

	random_shuffle(epoch.begin(), epoch.end());

	trainer.train(epoch);

	cout << "[DEBUG] Distance after: " << distance(network, epoch) << endl;

	if (!network.save(network_file))
	{
		cerr << "Unable to save network to " << network_file << endl;
		return 3;
	}
	cout << "Press return to exit" << endl;
	cin.get();
	return 0;*/
}

int main(int ac, char** av)
{
	if (ac < 4)
		return print_usage(av[0]);

	string dataset_folder(av[1]);
	unsigned int minibatch_size;
	{
		istringstream stream(av[2]);
		if (!(stream >> minibatch_size))
			return print_usage(av[0]);
	}
	string network_file(av[3]);

	return ocr_training(dataset_folder, minibatch_size, network_file, ac - 4, av + 4);
}