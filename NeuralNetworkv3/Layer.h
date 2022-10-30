#pragma once

#include "Matrix.h"
#include "Activation.h"
#include <memory>
#include "Utility.h"

namespace net
{
	class Layer
	{
	public:
		Layer(Layer& in, math::DMatrix biases, size_t n_nodes, double wmin = -1.0, double wmax = 1.0);
		Layer(size_t n_nodes);

		math::DMatrix Forward(const math::DMatrix& input, const actf::Activation& activation, bool start = false);
	public:
		const math::DMatrix& GetWeights() const;
		void SetWeights(const math::DMatrix& value);

		const math::DMatrix& GetBiases() const;
		void SetBiases(const math::DMatrix& value);

		const math::DMatrix& GetWeightedInputs() const;
		const math::DMatrix& GetOutputs() const;
	private:
		size_t n_nodes = 0;
		math::DMatrix weights; // inputs x outputs
		math::DMatrix biases;
		math::DMatrix weightedInputs{};
		math::DMatrix outputs{};
	};
}