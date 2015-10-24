#include <sstream>
#include <iostream>
#include <cmath>
#include <string>

#ifdef WIN32
# include <direct.h>
# include "dirent.h"
# define chdir _chdir
# define strerror _strerror
#else
# include <dirent.h>
#endif

#include "ocr_utils.hpp"
#include "Trainer.hpp"

using namespace nn;
using namespace std;

char* _strerror(int err)
{
	static char buffer[256];

	strerror_s(buffer, err);
	return buffer;
}

int print_usage(const char* exeName)
{
	cout << "Usage: " << exeName << " dataset_folder minibatch_size network_file [network_layer_size...]" << endl;
	return 1;
}

double distance(const NeuralNetwork& network, const string& dataset_folder)
{
	double distance = 0;
	unsigned int inputCount = 0;

	DIR* dir = opendir(dataset_folder.c_str());
	struct dirent* ent;

	while ((ent = readdir(dir)))
	{
		NeuralFeed expected(std::move(ocr::getExpectedOutput(ent->d_name)));

		if (!expected.empty())
		{
			NeuralFeed output(std::move(network.update(std::move(ocr::getInput(dataset_folder, ent->d_name)))));

			++inputCount;
			double tmp = 0;
			for (size_t i = 0; i < expected.size(); i++)
				tmp += abs(expected[i] - output[i]);
			distance += tmp / expected.size();
		}
	}
	closedir(dir);
	if (inputCount == 0)
		++inputCount;
	return distance / inputCount;
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

	cout << "[DEBUG] Distance before: " << distance(network, dataset_folder) << endl;

	DIR* dir;
	if (!(dir = opendir(dataset_folder.c_str())))
	{
		cerr << "Unable to open " << dataset_folder << ": " << strerror(errno) << endl;
		return 3;
	}

	struct dirent* ent;

	while ((ent = readdir(dir)))
	{
		NeuralFeed output(std::move(ocr::getExpectedOutput(ent->d_name)));

		if (!output.empty())
		{
			NeuralFeed input(std::move(ocr::getInput(dataset_folder, ent->d_name)));

			trainer.feed(std::make_pair(input, output));
		}
	}
	closedir(dir);

	trainer.flush();

	cout << "[DEBUG] Distance after: " << distance(network, dataset_folder) << endl;

	if (!network.save(network_file))
	{
		cerr << "Unable to save network to " << network_file << endl;
		return 3;
	}
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