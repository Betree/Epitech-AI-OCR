//
// Created by piouffb on 10/4/15.
//

#include <iostream>
#include "NeuralNetwork.hpp"

using namespace nn;

int main()
{
    std::vector<unsigned int> layersDefinitions;
    layersDefinitions.push_back(12);

    NeuralNetwork network(8, 2, layersDefinitions);

    // Feed network with values between 0 and 20
    for (unsigned char cInput = 1; cInput < 20; ++cInput)
    {
        std::vector<double> feedingData;

        // Decompose char into 8 boules
        for (int cBinaryComparator = 128; cBinaryComparator > 0; cBinaryComparator /= 2)
        {
            feedingData.push_back((cInput & cBinaryComparator) != 0);
        }

        std::vector<double> updateResult = network.update(feedingData);
        std::cout << "Testing " << (int)cInput << std::endl;
        std::cout << "Pair probability : " << updateResult[0] << std::endl;
        std::cout << "Impair probability : " << updateResult[1] << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }
}