#pragma once

#include "Activation.h"
#include "Cost.h"
#include "Utility.h"
#include "Layer.h"
#include <string>
#include <memory>

namespace net
{	
	class Network
	{
	public:
		Network(std::vector<size_t> layer_c, cost::Cost<double>* cost,
			std::unique_ptr<actf::Activation> hiddenActiv,
			std::unique_ptr<actf::Activation> outputActiv,
			double bias = 0.0);
		Network(std::string path);
	public:
		void CalculateOutputs(util::DataPoint<double>& dp);
		void CalculateOutputs(std::vector<util::DataPoint<double>>& batch);

		math::DMatrix Feed(math::DMatrix input);
		void Learn(std::vector<util::DataPoint<double>>& batch, double learnRate);

		void Save(std::string path) const;
		void Load(std::string path);
	private:
		void ApplyGradients(double learnRate);
		void ClearGradients();
		void UpdateGradients(size_t layer_i, math::DMatrix nodeValues); // accumulates the gradients and use the average of the gradients when being applied
		void GetGradients(util::DataPoint<double>& dp);

		math::DMatrix OutputLayerValues(util::DataPoint<double>& dp) const;
		math::DMatrix HiddenLayerValues(size_t layer_i, math::DMatrix nodeValues) const;
	private:
		std::vector<Layer> layers;

		std::vector<math::DMatrix> weight_grad;
		std::vector<math::DMatrix> bias_grad;

		std::unique_ptr<actf::Activation> hiddenActiv;
		std::unique_ptr<actf::Activation> outputActiv;

		cost::Cost<double>* cost;

		std::vector<size_t> layer_c;
		size_t n_layers;
	};
}