#pragma once

#include <random>
#include "Matrix.h"
#include <memory>

#define SEED 4252452

namespace util
{
	template<typename T>
	struct DataPoint
	{
		DataPoint(math::Matrix<T> input, math::Matrix<T> expected) : input(input), expected(expected) {};
		DataPoint(math::Matrix<T> input, math::Matrix<T> expected, math::Matrix<T> output) : output(output), expected(expected), input(input) {};
		DataPoint() = default;

		math::Matrix<T> input;
		math::Matrix<T> expected;
		math::Matrix<T> output;
		T label;
	};

	inline std::random_device _rd;
	inline std::mt19937 _rng(SEED);
	template<typename T, typename distr>
	inline T Random(distr dist)
	{
		return dist(_rng);
	}

	template<typename T>
	inline double Accuracy(std::vector<util::DataPoint<T>> data)
	{
		int correct = 0;
		for (util::DataPoint<T> dp : data)
		{
			int chosen = (int)(std::max_element(dp.output.begin(), dp.output.end()) - dp.output.begin());
			if (chosen == (int)dp.label)
			{
				correct++;
			}
		}
		return (double)correct / data.size();
	}
}