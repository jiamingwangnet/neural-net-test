#pragma once

#include "Cost.h"

namespace net
{
	namespace cost
	{
		template<typename T>
		class MSE : public Cost<T>
		{
			T Calculate(T predicted, T expected) const override
			{
				return (predicted - expected) * (predicted - expected);
			}

			T Calculate(util::DataPoint<T> dp) const override
			{
				T sum = 0.0;
				for (size_t i = 0; i < dp.expected.GetSize(); i++)
				{
					sum += this->Calculate(dp.output[i], dp.expected[i]);
				}
				return sum;
			}

			T Calculate(std::vector<util::DataPoint<T>> batch) const override
			{
				T sum = 0.0;
				for (util::DataPoint<T> dp : batch)
				{
					sum += this->Calculate(dp);
				}
				return sum / batch.size();
			}

			T Calculate(std::vector<std::vector<util::DataPoint<T>>> dataSet) const override
			{
				T sum = 0.0;
				for (std::vector<util::DataPoint<T>> batch : dataSet)
				{
					for(util::DataPoint<T> dp : batch)
					{
						sum += this->Calculate(dp);
					}
				}
				return sum / (dataSet.size() * dataSet[0].size());
			}

			T Derivative(T predicted, T expected) const override
			{
				return 2.0 * (predicted - expected);
			}

			math::Matrix<T> Derivative(util::DataPoint<T> dp) const override
			{
				math::Matrix<T> res{dp.expected.GetRows(), dp.expected.GetColumns()};
				for (size_t i = 0; i < dp.expected.GetSize(); i++)
				{
					res[i] = this->Derivative(dp.output[i], dp.expected[i]);
				}
				return res;
			}
		};

		template<typename T>
		class CrossEntropy : public Cost<T>
		{
			T Calculate(T predicted, T expected) const override
			{
				T v = expected == 1.0 ? -std::log(predicted) : -std::log(1.0 - predicted);
				return std::isnan(v) ? 0.0 : v;
			}

			T Calculate(util::DataPoint<T> dp) const override
			{
				T sum = 0.0;
				for (size_t i = 0; i < dp.expected.GetSize(); i++)
				{
					sum += this->Calculate(dp.output[i], dp.expected[i]);
				}
				return sum;
			}

			T Calculate(std::vector<util::DataPoint<T>> batch) const override
			{
				T sum = 0.0;
				for (util::DataPoint<T> dp : batch)
				{
					sum += this->Calculate(dp);
				}
				return sum / batch.size();
			}

			T Calculate(std::vector<std::vector<util::DataPoint<T>>> dataSet) const override
			{
				T sum = 0.0;
				for (std::vector<util::DataPoint<T>> batch : dataSet)
				{
					for (util::DataPoint<T> dp : batch)
					{
						sum += this->Calculate(dp);
					}
				}
				return sum / (dataSet.size() * dataSet[0].size());
			}

			T Derivative(T predicted, T expected) const override
			{
				if (predicted == 0.0 || predicted == 1.0)
				{
					return 0.0;
				}
				return (-predicted + expected) / (predicted * (predicted - 1));
			}

			math::Matrix<T> Derivative(util::DataPoint<T> dp) const override
			{
				math::Matrix<T> res{ dp.expected.GetRows(), dp.expected.GetColumns() };
				for (size_t i = 0; i < res.GetSize(); i++)
				{
					res[i] = this->Derivative(dp.output[i], dp.expected[i]);
				}
				return res;
			}
		};
	}
}