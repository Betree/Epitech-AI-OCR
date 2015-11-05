#include <sstream>
#include <iostream>
#include <cmath>
#include <mutex>
#include <thread>
#include <algorithm>
#include <string.h>

#ifdef WIN32
# include <direct.h>
# include "dirent.h"
# define strerror _strerror
#else
# include <dirent.h>
#endif

#include "opencv2/opencv.hpp"
#include "ocr_utils.hpp"
#include "Trainer.hpp"
#include "cvplot.h"

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
		for (size_t i = 0; i < expected.size(); i++)
			tmp += abs(expected[i] - output[i]);
		distance += tmp / expected.size();
	}

	if (inputCount == 0)
		++inputCount;
	return distance / inputCount;
}

class PlottingTrainer
{
private:
	static const int maxSize = 100;

	const string& _fileName;
	NeuralNetwork& _network;
	Trainer& _trainer;
	Trainer::Epoch& _epoch;

	mutex _distanceMutex;
	list<double> _distances;

public:
	PlottingTrainer(const string& fileName, NeuralNetwork& network, Trainer& trainer, Trainer::Epoch& epoch)
		: _fileName(fileName), _network(network), _trainer(trainer), _epoch(epoch), _distances()
	{
		this->_distances.push_back(distance(this->_network, this->_epoch) * 1000000.0);
	}

	void train()
	{
		this->_trainer.train(this->_epoch);
		this->_network.save(this->_fileName);

		double d = distance(this->_network, this->_epoch) * 1000000.0;

		this->_distanceMutex.lock();
		this->_distances.push_back(d);
		while (this->_distances.size() > maxSize)
			this->_distances.pop_front();
		this->_distanceMutex.unlock();

		this->DISPLAY();
	}

	void DISPLAY()
	{
		CvPlot::clear("Distance");
		this->_distanceMutex.lock();
		CvPlot::plot("Distance", this->_distances.begin(), this->_distances.size(), 1, 60, 255, 60);
		this->_distanceMutex.unlock();
	}
};

PlottingTrainer* pt;

bool keepTraining = true;

void trainLoop()
{
	while (keepTraining)
	{
		pt->train();
	}
}

thread trainingThread;

void startThread()
{
	trainingThread = thread(&trainLoop);
}

void stopThread()
{
	keepTraining = false;
	trainingThread.join();
}

int ocr_training(const string& dataset_folder, unsigned int minibatch_size, const string& network_file, int ac, char** av)
{
	NeuralNetwork network;
	Trainer trainer(&network);

	trainer.setMiniBatchSize(minibatch_size);
	if (ac > 0)
		network = ocr::fromArgv(ac, av);
	else if (!network.load(network_file))
	{
		cerr << "Unable to load network from " << network_file << endl;
		return 2;
	}


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

		if (filename.length() >= 4 && !filename.compare(filename.length() - 4, 4, ".bmp"))
		{
			cout << "Processing " << filename << "... ";
			NeuralFeed output(std::move(ocr::getExpectedOutput(filename)));

			if (!output.empty()) {
				NeuralFeed input(std::move(ocr::getInput(dataset_folder, filename)));

				epoch.push_back(Trainer::InputOutputPair(input, output));
			}
			cout << "Done!" << endl;
		}
	}

	closedir(dir);

	std::random_shuffle(epoch.begin(), epoch.end());

	pt = new PlottingTrainer(network_file, network, trainer, epoch);

	ac += 4;
	av -= 4;

	startThread();

	pt->DISPLAY();
	while (cv::waitKey(0) != -1)
	{
	}

	stopThread();

	//cout << "[DEBUG] Distance before: " << distance(network, epoch) << endl;

	//cout << "Training start." << endl;

	//trainer.train(epoch);

	//cout << "Training done." << endl;

	//cout << "[DEBUG] Distance after: " << distance(network, epoch) << endl;

	//if (!network.save(network_file))
	//{
	//	cerr << "Unable to save network to " << network_file << endl;
	//	return 3;
	//}
	return 0;
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