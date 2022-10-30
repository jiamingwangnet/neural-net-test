#pragma once

#include "Matrix.h"
#include "Utility.h"

namespace net
{
	namespace actf
	{
		enum class ACTIVATION_TYPE
		{
			SIGMOID,
			RELU,
			SOFTMAX
		};

		class Activation
		{
		public:
			virtual math::Matrix<double> Activate(math::Matrix<double> nodes) const = 0;
			virtual math::Matrix<double> Derivative(math::Matrix<double> nodes) const = 0;
			virtual ACTIVATION_TYPE GetType() const = 0;
		};
	}
}