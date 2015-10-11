#ifndef TRAINER_HPP_
#define	TRAINER_HPP_

#include <list>
#include "NeuralNetwork.hpp"

namespace nn
{
	class Trainer
	{
	public:
		typedef std::pair<NeuralFeed, NeuralFeed> InputOutputPair;

		Trainer(NeuralNetwork* network);
		~Trainer();
		Trainer(const Trainer& other);
		Trainer& operator=(const Trainer& other);

		unsigned int getMiniBatchSize() const;
		void setMiniBatchSize(unsigned int value);

		void train(const std::vector<InputOutputPair>& epoch);

		void feed(const InputOutputPair& input);
		void flush();

	private:
		NeuralNetwork* _network;
		unsigned int _miniBatchSize;
		std::vector<InputOutputPair> _miniBatch;
	};
}
#endif	// TRAINER_HPP_