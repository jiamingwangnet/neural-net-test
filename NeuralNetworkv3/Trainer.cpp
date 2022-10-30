#include "Trainer.h"

util::Trainer::Trainer(std::vector<DataPoint<double>>& data, size_t batchSize, float trainPercent)
{
	size_t trainSize = (size_t)std::floorf((float)data.size() * trainPercent);
	size_t testSize = data.size() - trainSize;

	std::vector<DataPoint<double>> batch;

	for (size_t i = 0; i < trainSize; i++)
	{
		batch.push_back(data[i]);
		if (i != 0 && i % batchSize == 0)
		{
			trainBatches.push_back(batch);
			batch.clear();
		}
	}
	if (batch.size() != 0)
	{
		trainBatches.push_back(batch);
		batch.clear();
	}

	for (size_t i = 0; i < testSize; i++)
	{
		batch.push_back(data[i]);
		if (i != 0 && i % batchSize == 0)
		{
			testBatches.push_back(batch);
			batch.clear();
		}
	}
	if (batch.size() != 0)
	{
		testBatches.push_back(batch);
		batch.clear();
	}
}

void util::Trainer::Train(net::Network& net, double learnRate, size_t index)
{
	net.Learn(trainBatches[index], learnRate);
}

void util::Trainer::Test(net::Network& net, size_t index)
{
	net.CalculateOutputs(testBatches[index]);
}

void util::Trainer::Test(net::Network& net)
{
	for (auto& batch : testBatches)
	{
		net.CalculateOutputs(batch);
	}
}

const std::vector<std::vector<util::DataPoint<double>>>& util::Trainer::GetTrainBatches() const
{
	return trainBatches;
}

const std::vector<std::vector<util::DataPoint<double>>>& util::Trainer::GetTestBatches() const
{
	return testBatches;
}
