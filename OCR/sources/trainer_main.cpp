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

	if (chdir(dataset_folder.c_str()))
	{
		cerr << "Can not go to " << dataset_folder << ": " << strerror(errno) << endl;
		return 4;
	}

	DIR* dir = opendir(".");
	struct dirent* ent;

	while ((ent = readdir(dir)))
	{
		NeuralFeed output(std::move(ocr::getExpectedOutput(ent->d_name)));

		if (!output.empty())
		{
			NeuralFeed input(std::move(ocr::getInput(ent->d_name)));

			trainer.feed(std::make_pair(input, output));
		}
	}
	closedir(dir);

	trainer.flush();
	if (network.save(network_file))
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