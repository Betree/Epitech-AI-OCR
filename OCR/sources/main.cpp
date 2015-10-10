//
// Created by piouffb on 10/4/15.
//

#include <random>
#include <iostream>
#include <cmath>
#include <string>
#include "NeuralNetwork.hpp"
#include "Trainer.hpp"

using namespace nn;
using namespace std;

int main()
{
    NeuralNetwork network(8, 2, { 4 });
	Trainer trainer(&network);

	// Generate test epoch.
	const unsigned int epochSize = 1000;
	vector<Trainer::InputOutputPair> epoch;
	epoch.reserve(epochSize);
	default_random_engine generator;
	uniform_int_distribution<int> distribution(numeric_limits<char>::min(), numeric_limits<char>::max());
	for (size_t i = 0; i < epochSize; i++)
	{
		unsigned char cInput = distribution(generator);
		vector<double> feedingData;

		// Decompose char into 8 boules
		for (int cBinaryComparator = 128; cBinaryComparator > 0; cBinaryComparator /= 2)
		{
			feedingData.push_back((cInput & cBinaryComparator) != 0);
		}
		epoch.push_back(Trainer::InputOutputPair(feedingData, { (double)((cInput % 2) == 0), (double)((cInput % 2) != 0) }));
	}

	string line;
	while (cin)
	{
		char cInput;

		cout << "Feed value: ";
		if (cin >> cInput)
		{
			cin.clear();
			cin.ignore((streamsize)numeric_limits<streamsize>::max, '\n');

			vector<double> feedingData;
			// Decompose char into 8 boules
			for (int cBinaryComparator = 128; cBinaryComparator > 0; cBinaryComparator /= 2)
			{
				feedingData.push_back((cInput & cBinaryComparator) != 0);
			}

			vector<double> updateResult = network.update(feedingData);
			cout << "Testing " << (int)cInput << endl;
			cout << "Pair probability : " << updateResult[0] << endl;
			cout << "Impair probability : " << updateResult[1] << endl;
			cout << "----------------------------------------" << endl;

			cout << "Press return to feed an epoch or Ctrl+D to stop...";
			if (getline(cin, line))
			{
				trainer.train(epoch);
			}
		}
	}
	return 0;

    // Feed network with values between 0 and 20
 //   for (unsigned char cInput = 1; cInput < 20; ++cInput)
 //   {
 //       std::vector<double> feedingData;

 //       // Decompose char into 8 boules
 //       for (int cBinaryComparator = 128; cBinaryComparator > 0; cBinaryComparator /= 2)
 //       {
 //           feedingData.push_back((cInput & cBinaryComparator) != 0);
 //       }

 //       std::vector<double> updateResult = network.update(feedingData);
 //       std::cout << "Testing " << (int)cInput << std::endl;
 //       std::cout << "Pair probability : " << updateResult[0] << std::endl;
 //       std::cout << "Impair probability : " << updateResult[1] << std::endl;
 //       std::cout << "----------------------------------------" << std::endl;
 //   }
	//std::cin.get();
}