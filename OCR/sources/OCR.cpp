#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <cctype>

#ifdef _WIN32
# include <direct.h>
# include "dirent.h"
# define GetCurrentDir _getcwd
# define chdir _chdir
#else
# include <unistd.h>
# include <dirent.h>
# define GetCurrentDir getcwd
#endif

#include "OCR.h"
#include "ocr_utils.hpp"

using namespace nn;
using namespace std;

template<typename T>
static string toString(const T& value)
{
	ostringstream stream;

	stream << value;
	return stream.str();
}

OCR::OCR()
	: _exit(false)
{
	this->_commands["help"] = &OCR::help;
	this->_commands["exit"] = &OCR::exit;
	this->_commands["pwd"] = &OCR::pwd;
	this->_commands["cd"] = &OCR::cd;
	this->_commands["ls"] = &OCR::ls;
	this->_commands["load"] = &OCR::loadNetwork;
	this->_commands["create"] = &OCR::createNetwork;
	this->_commands["upgrade"] = &OCR::upgradeNetwork;
	this->_commands["save"] = &OCR::saveNetwork;
	this->_commands["test_letter"] = &OCR::testLetterFile;
	this->_commands["test_directory"] = &OCR::testDirectory;

	this->_env["OCR_INPUT_NUMBER"] = toString(OCR_INPUT_NUMBER);
	this->_env["OCR_OUTPUT_NUMBER"] = toString(OCR_OUTPUT_NUMBER);
}

OCR::~OCR()
{
}

OCR::Args OCR::parseArgs(const std::string& line) const
{
	OCR::Args args;
	char quote = -1;
	bool wasSpace = true;

	for (char c : line)
	{
		if (quote != -1)
		{
			if (c == quote)
				quote = -1;
			else
			{
				if (wasSpace)
					args.push_back(string());
				args.back().push_back(c);
				wasSpace = false;
			}
		}
		else if (iswspace(c))
		{
			wasSpace = true;
		}
		else if (c == '\'' || c == '"')
		{
			quote = c;
		}
		else
		{
			if (wasSpace)
				args.push_back(string());
			args.back().push_back(c);
			wasSpace = false;
		}
	}
	for (string& arg : args)
	{
		if (arg.size() > 0 && arg[0] == '$')
		{
			const auto& found(this->_env.find(arg.substr(1)));

			if (found != this->_env.end())
				arg = found->second;
		}
	}
	return args;
}

void OCR::start()
{
	while (!this->_exit && cin)
	{
		string line;
		cout << "> " << flush;
		if (getline(cin, line))
		{
			Args args(this->parseArgs(line));

			if (args.size() > 0)
			{
				const auto& found(this->_commands.find(args[0]));

				if (found == this->_commands.end())
				{
					cerr << "Unknown command " << args[0] << ", try help for help." << endl;
				}
				else
				{
					try
					{
						(this->*(found->second))(args);
					}
					catch (const std::exception& e)
					{
						cerr << "Command thrown exception: " << e.what() << endl;
					}
				}
			}
		}
	}
}

nn::NeuralNetwork& OCR::getNetwork()
{
	return this->_network;
}

const nn::NeuralNetwork& OCR::getNetwork() const
{
	return this->_network;
}

int OCR::exit(const Args& args)
{
	this->_exit = true;
	return 0;
}

int OCR::help(const Args& args)
{
	cout << "List of available commands:" << endl;
	for (const auto& keyValue : this->_commands)
		cout << keyValue.first << endl;
	return 0;
}

int OCR::pwd(const Args& args)
{
	if (args.size() > 1)
	{
		cerr << "Usage: " << args[0] << endl;
		return 1;
	}
	char cCurrentPath[FILENAME_MAX];

	if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath)))
	{
		cerr << "Failed to retrieve current directory" << endl;
		return errno;
	}

	cout << cCurrentPath << endl;
	return 0;
}

int OCR::cd(const Args& args)
{
	if (args.size() != 2)
	{
		cerr << "Usage: " << args[0] << " path" << endl;
		return 1;
	}
	if (chdir(args[1].c_str()))
	{
		cerr << "Failed to go to: " << args[1] << endl;
		return errno;
	}
	return 1;
}

int OCR::ls(const Args& args)
{
	if (args.size() > 2)
	{
		cerr << "Usage: " << args[0] << " [directory]" << endl;
		return 1;
	}
	const char* dirName = (args.size() == 1 ? "." : args[1].c_str());

	DIR* dir;
	struct dirent* ent;

	if (!(dir = opendir(dirName)))
	{
		cerr << "Unable to open " << dirName << endl;
		return 3;
	}
	while ((ent = readdir(dir)))
	{
		cout << ent->d_name;
		if (ent->d_type == DT_DIR)
			cout << '/';
		cout << endl;
	}
	return 0;
}

int OCR::loadNetwork(const Args& args)
{
	if (args.size() != 2)
	{
		cerr << "Usage: " << args[0] << " network_file" << endl;
		return 1;
	}
	if (!this->getNetwork().load(args[1]))
	{
		cerr << "Failed to load " << args[1] << endl;
		return 2;
	}
	return 0;
}

static int networkFromArgs(const std::vector<std::string>& args, NeuralNetwork& network)
{
	if (args.size() < 3)
	{
		cerr << "Usage: " << args[0] << " input_size [layer_size...] output_size" << endl;
		return 1;
	}
	unsigned int input = 0;
	unsigned int output;
	vector<unsigned int> layers;

	for (size_t i = 1; i < args.size(); i++)
	{
		istringstream stream(args[i]);
		int value;

		if (!(stream >> value) || value == 0)
		{
			cerr << "Bad formated value " << args[i] << endl;
			return 2;
		}
		else if (input == 0)
			input = value;
		else
			layers.push_back(value);
	}
	output = layers.back();
	layers.pop_back();
	NeuralNetwork created(input, output, layers);

	if (created.size() == 0)
	{
		cerr << "Failed to create network" << endl;
		return 3;
	}
	network = created;
	return 0;

}

int OCR::createNetwork(const Args& args)
{
	return networkFromArgs(args, this->getNetwork());
}

int OCR::upgradeNetwork(const Args& args)
{
	NeuralNetwork tmp;
	int ret = networkFromArgs(args, tmp);

	if (ret == 0)
	{
		for (size_t i = 0; i < min(this->getNetwork().size(), tmp.size()); i++)
		{
			for (size_t j = 0; j < min(this->getNetwork()[i].size(), tmp[i].size()); j++)
			{
				for (size_t k = 0; k < min(this->getNetwork()[i][j].size(), tmp[i][j].size()); k++)
				{
					tmp[i][j].setWeight(k, this->getNetwork()[i][j].getWeight(k));
				}
				tmp[i][j].setBias(this->getNetwork()[i][j].getBias());
			}
		}
		this->getNetwork() = tmp;
	}
	return ret;
}

int OCR::saveNetwork(const Args& args)
{
	if (args.size() != 2)
	{
		cerr << "Usage: " << args[0] << " filename" << endl;
		return 1;
	}
	if (this->getNetwork().size() == 0)
	{
		cerr << "A network must be loaded or created before being saved" << endl;
		return 2;
	}
	this->getNetwork().save(args[1]);
	return 0;
}

static string getPathDirectory(const string& path)
{
	size_t last = path.find_last_of('/');

	if (last == string::npos)
		return "./";
	return path.substr(0, last + 1);
}

static string getPathFileName(const string& path)
{
	size_t last = path.find_last_of('/');

	if (last == string::npos)
		return path;
	return path.substr(last + 1);
}

int OCR::testLetterFile(const Args& args)
{
	if (args.size() != 2)
	{
		cerr << "Usage: " << args[0] << " filename" << endl;
		return 1;
	}
	if (this->getNetwork().size() == 0)
	{
		cerr << "A network must be loaded or created before being saved" << endl;
		return 2;
	}
	ocr::ocr_test(this->getNetwork(), getPathDirectory(args[1]), getPathFileName(args[1]));
	return 0;
}

int OCR::testDirectory(const Args & args)
{
	if (args.size() != 2)
	{
		cerr << "Usage: " << args[0] << " directory" << endl;
		return 1;
	}
	if (this->getNetwork().size() == 0)
	{
		cerr << "A network must be loaded or created before being saved" << endl;
		return 2;
	}
	DIR* dir;
	struct dirent* ent;
	unsigned int total = 0;
	unsigned int success = 0;

	if (!(dir = opendir(args[1].c_str())))
	{
		cerr << "Unable to open " << args[1] << endl;
		return 3;
	}

	while ((ent = readdir(dir)))
	{
		string filename(ent->d_name);

		if (filename.length() >= 4 && !filename.compare(filename.length() - 4, 4, ".bmp"))
		{
			++total;
			success += ocr::ocr_test(this->getNetwork(), args[1], filename);
		}
	}
	closedir(dir);
	if (total > 0)
		cout << "Succes rate: " << ((double)success * 100.0 / (double)total) << '%' << endl;
	return 0;
}