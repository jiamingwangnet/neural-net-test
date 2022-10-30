#pragma once

#include "Activation.h"

namespace net
{
	namespace actf
	{
		class Sigmoid : public Activation
		{
		public:
			math::Matrix<double> Activate(math::Matrix<double> nodes) const override
			{
				math::Matrix<double> res{ nodes.GetRows(), nodes.GetColumns() };
				for (size_t i = 0; i < nodes.GetSize(); i++)
				{
					res[i] = 1.0 / (1.0 + std::exp(-nodes[i]));
				}
				return res;
			}

			math::Matrix<double> Derivative(math::Matrix<double> nodes) const override
			{
				math::Matrix<double> res{ nodes.GetRows(), nodes.GetColumns() };
				for (size_t i = 0; i < nodes.GetSize(); i++)
				{
					double activation = 1.0 / (1.0 + std::exp(-nodes[i]));
					res[i] = activation * (1.0 - activation);
				}
				return res;
			}

			ACTIVATION_TYPE GetType() const override
			{
				return ACTIVATION_TYPE::SIGMOID;
			}
		};

		class ReLU : public Activation
		{
			math::Matrix<double> Activate(math::Matrix<double> nodes) const override
			{
				math::Matrix<double> res{ nodes.GetRows(), nodes.GetColumns() };
				for (size_t i = 0; i < nodes.GetSize(); i++)
				{
					res[i] = std::max(0.0, nodes[i]);
				}
				return res;
			}

			math::Matrix<double> Derivative(math::Matrix<double> nodes) const override
			{
				math::Matrix<double> res{ nodes.GetRows(), nodes.GetColumns() };
				for (size_t i = 0; i < nodes.GetSize(); i++)
				{
					res[i] = nodes[i] <= 0.0 ? 0.0 : 1.0;
				}
				return res;
			}

			ACTIVATION_TYPE GetType() const override
			{
				return ACTIVATION_TYPE::RELU;
			}
		};

		class Softmax : public Activation
		{
			math::Matrix<double> Activate(math::Matrix<double> nodes) const override
			{
				math::Matrix<double> res{ nodes.GetRows(), nodes.GetColumns() };
				double expSum = 0.0;
				for (double& v : nodes)
				{
					expSum += std::exp(v);
				}
				for (size_t i = 0; i < nodes.GetSize(); i++)
				{
					res[i] = std::exp(nodes[i]) / expSum;
				}
				return res;
			}

			math::Matrix<double> Derivative(math::Matrix<double> nodes) const override
			{
				math::Matrix<double> res{ nodes.GetRows(), nodes.GetColumns() };

				double sum = 0.0;
				for (double& v : nodes)
				{
					sum += std::exp(v);
				}

				for (size_t i = 0; i < nodes.GetSize(); i++)
				{
					double ex = std::exp(nodes[i]);

					res[i] = (ex * sum - ex * ex) / (sum * sum);
				}
				return res;
			}

			ACTIVATION_TYPE GetType() const override
			{
				return ACTIVATION_TYPE::SOFTMAX;
			}
		};

		inline std::unique_ptr<Activation> GetActivation(ACTIVATION_TYPE type)
		{
			switch (type)
			{
			case ACTIVATION_TYPE::SIGMOID:
				return std::make_unique<Sigmoid>();
			case ACTIVATION_TYPE::RELU:
				return std::make_unique<ReLU>();
			case ACTIVATION_TYPE::SOFTMAX:
				return std::make_unique<Softmax>();
			default:
				return nullptr;
			}
		}
	}
}