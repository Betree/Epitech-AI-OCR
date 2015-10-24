#pragma once

#include "NeuralNetwork.hpp"

#define OCR_INPUT_NUMBER 256 // Debug
#define OCR_OUTPUT_NUMBER 256 // Debug

namespace ocr
{
	nn::NeuralNetwork fromArgv(unsigned int ac, char** argv);

	nn::NeuralFeed getInput(const std::string& folder, const std::string& fileName);

	char getCharFromOutput(const nn::NeuralFeed& output);

	nn::NeuralFeed getExpectedOutput(const std::string& fileName);
	char getExpectedChar(const std::string& fileName);

	char getChar(const nn::NeuralNetwork& network, const std::string& directory, const std::string& filename);
}