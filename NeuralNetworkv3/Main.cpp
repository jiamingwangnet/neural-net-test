#include "Network.h"
#include "ActivationFuncs.h"
#include "CostFuncs.h"
#include "Trainer.h"
#include <iostream>
#include <conio.h>

bool isSafe(int x, int y)
{
	return ((x / 2) + (x / 2) * (x / 2)) < y && y < (- 6 * x * x + 10 * x * x * x);
}

int main()
{
	using namespace net;
	using std::string;

	// {0, 1} unsafe
	// {1, 0} safe
	util::DataPoint<double> unsafe{ {{4.28, 2.87},1,2},{{0.0,1.0},1,2} };
	util::DataPoint<double> safe{ {{2.45, 5.5},1,2},{{1.0,0.0},1,2} };

	std::vector<util::DataPoint<double>> data;
	for (int i = 0; i < 20000; i++)
	{
		util::DataPoint<double> dp;

		int x = util::Random<int>(std::uniform_int_distribution<int>{0, 10});
		int y = util::Random<int>(std::uniform_int_distribution<int>{0, 10});

		dp.input = math::DMatrix{ {(double)x, (double)y}, 1, 2  };
		dp.expected = (isSafe(x, y) ? math::DMatrix{ {1.0, 0.0}, 1, 2 } : math::DMatrix{ {0.0, 1.0}, 1, 2 });

		data.push_back(dp);
	}

	cost::MSE<double> mse;

	Network network{ {2,3,2}, &mse, std::move(std::make_unique<actf::Sigmoid>()), std::move(std::make_unique<actf::Sigmoid>()) };

	util::Trainer trainer{ data, 100, 0.8f };

	for (size_t index = 0, i = 0;;index++,i++)
	{
		trainer.Train(network, 0.05, index);
		network.CalculateOutputs(safe);
		network.CalculateOutputs(unsafe);
		
		std::cout << "---------------------------------------------------\n";
		std::cout << "epoch: " << i << '\n';
		std::cout << "0 | predicted: safe: " << (safe.output[0] * 100.0) << "% unsafe: " << (safe.output[1] * 100.0) << "% expected: safe: " << (safe.expected[0] * 100.0) << "% unsafe: " << (safe.expected[1] * 100.0) << '%' << '\n';
		std::cout << "1 | predicted: safe: " << (unsafe.output[0] * 100.0) << "% unsafe: " << (unsafe.output[1] * 100.0) << "% expected: safe: " << (unsafe.expected[0] * 100.0) << "% unsafe: " << (unsafe.expected[1] * 100.0) << '%' << '\n';
		std::cout << "---------------------------------------------------\n";

		if (index == trainer.GetTrainBatches().size() - 1)
		{
			index = -1;
		}

		if (_kbhit())
			break;
	}

	string name;
	std::cout << "file name: ";
	std::cin >> name;

	network.Save(name + ".txt");

	std::cout << "\n saved to " << name << ".txt\n";

	return 0;
}