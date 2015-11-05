#pragma once

#include <vector>
#include <string>
#include "NeuralNetwork.hpp"

#define VERTICAL_DENSITY_POINTS 10
#define HORIZONTAL_DENSITY_POINTS 10

#define OCR_INPUT_NUMBER VERTICAL_DENSITY_POINTS + HORIZONTAL_DENSITY_POINTS + 4
#define OCR_OUTPUT_NUMBER 128

namespace ocr
{
	nn::NeuralNetwork fromArgv(const std::vector<std::string>& args);
	nn::NeuralNetwork fromArgv(unsigned int ac, char** argv);

	nn::NeuralFeed getInput(const std::string& folder, const std::string& fileName);

	char getCharFromOutput(const nn::NeuralFeed& output);

	nn::NeuralFeed getExpectedOutput(const std::string& fileName);
	char getExpectedChar(const std::string& fileName);

	char getChar(const nn::NeuralNetwork& network, const std::string& directory, const std::string& filename);

	bool ocr_test(const nn::NeuralNetwork& network, const std::string& dir, const std::string& file);
}