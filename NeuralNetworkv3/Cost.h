#pragma once

#include "Utility.h"

namespace net
{
	namespace cost
	{
		template<typename T>
		class Cost
		{
		public:
			virtual T Calculate(T predicted, T expected) const = 0;
			virtual T Calculate(util::DataPoint<T> dp) const = 0;
			virtual T Calculate(std::vector<util::DataPoint<T>> batch) const = 0;
			virtual T Calculate(std::vector<std::vector<util::DataPoint<T>>> dataSet) const = 0;

			virtual T Derivative(T predicted, T expected) const = 0;
			virtual math::Matrix<T> Derivative(util::DataPoint<T>) const = 0; // gets the derivative for each row/column
		};
	}
}