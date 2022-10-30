#pragma once

#include <vector>
#include "Network.h"

namespace util
{
	class Trainer
	{
	public:
		Trainer(std::vector<DataPoint<double>>& data, size_t batchSize, float trainPercent);
	public:
		void Train(net::Network& net, double learnRate, size_t index);
		void Test(net::Network& net, size_t index);
		void Test(net::Network& net);

		const std::vector<std::vector<DataPoint<double>>>& GetTrainBatches() const;
		const std::vector<std::vector<DataPoint<double>>>& GetTestBatches() const;
	private:
		std::vector<std::vector<DataPoint<double>>> trainBatches;
		std::vector<std::vector<DataPoint<double>>> testBatches;
	};
}