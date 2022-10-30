#include "Layer.h"

net::Layer::Layer(Layer& in, math::DMatrix biases, size_t n_nodes, double wmin, double wmax)
	: biases(biases), n_nodes(n_nodes), weights(in.n_nodes, n_nodes)
{
	for (double& w : weights)
	{
		w = util::Random<double>(std::uniform_real_distribution<double>(wmin, wmax)) / std::sqrt((double)in.n_nodes);
	}
}

net::Layer::Layer(size_t n_nodes)
	: n_nodes(n_nodes)
{}

math::DMatrix net::Layer::Forward(const math::DMatrix& input, const actf::Activation& activation, bool start)
{
	if (start)
	{
		outputs = input;
		return input;
	}

	weightedInputs = input * weights + biases;
	outputs = activation.Activate(weightedInputs);
	return outputs;
}

const math::DMatrix& net::Layer::GetWeights() const
{
	return weights;
}

void net::Layer::SetWeights(const math::DMatrix& value)
{
	weights = value;
}

const math::DMatrix& net::Layer::GetBiases() const
{
	return biases;
}

void net::Layer::SetBiases(const math::DMatrix& value)
{
	biases = value;
}

const math::DMatrix& net::Layer::GetWeightedInputs() const
{
	return weightedInputs;
}

const math::DMatrix& net::Layer::GetOutputs() const
{
	return outputs;
}