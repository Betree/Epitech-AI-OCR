#pragma once

#include <string>
#include <map>
#include "NeuralNetwork.hpp"
#include "Trainer.hpp"

class OCR
{
public:
	OCR();
	~OCR();

	void start();

	nn::NeuralNetwork& getNetwork();
	const nn::NeuralNetwork& getNetwork() const;

private:
	typedef std::vector<std::string> Args;
	typedef int (OCR::*Command)(const Args& args);

	Args parseArgs(const std::string& line) const;

	int help(const Args& args);
	int exit(const Args& args);
	int pwd(const Args& args);
	int cd(const Args& args);
	int ls(const Args& args);
	int loadNetwork(const Args& args);
	int createNetwork(const Args& args);
	int upgradeNetwork(const Args& args);
	int saveNetwork(const Args& args);
	int testLetterFile(const Args& args);
	int testDirectory(const Args& args);
	int trainDirectory(const Args& args);

	bool _exit;

	std::map<std::string, Command> _commands;
	std::map<std::string, std::string> _env;

	nn::NeuralNetwork _network;
	nn::Trainer _trainer;
};

