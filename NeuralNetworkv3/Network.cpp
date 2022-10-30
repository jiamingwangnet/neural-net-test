#include "Network.h"
#include <fstream>
#include "ActivationFuncs.h"

net::Network::Network(std::vector<size_t> layer_c, cost::Cost<double>* cost, 
	std::unique_ptr<actf::Activation> hiddenActiv,
	std::unique_ptr<actf::Activation> outputActiv, 
	double bias)
	: layer_c(layer_c), cost(cost), hiddenActiv(std::move(hiddenActiv)), outputActiv(std::move(outputActiv))
{
	n_layers = layer_c.size();

	layers.emplace_back(layer_c[0]);
	for (auto c = layer_c.begin() + 1; c != layer_c.end() - 1; ++c)
	{
		size_t i = c - layer_c.begin();
		layers.emplace_back(layers[i - 1], math::DMatrix{ 1, *c, bias }, *c);
	}
	layers.emplace_back(layers[layers.size() - 1], math::DMatrix{ 1, layer_c[layer_c.size() - 1], bias }, layer_c[layer_c.size() - 1]);

	weight_grad.resize(layers.size());
	bias_grad.resize(layers.size());

	ClearGradients();
}

net::Network::Network(std::string path)
{
	Load(path);
}

void net::Network::CalculateOutputs(util::DataPoint<double>& dp)
{
	dp.output = Feed(dp.input);
}

void net::Network::CalculateOutputs(std::vector<util::DataPoint<double>>& batch)
{
	for (auto& dp : batch)
	{
		CalculateOutputs(dp);
	}
}

void net::Network::Save(std::string path) const
{
	std::ofstream out{ path };

	// n layers
	out << layer_c.size() << '\n';

	// layer sizes
	// l0 l1 l2 ...
	for (const size_t& c : layer_c)
	{
		out << c << ' ';
	}
	out << '\n';

	// activation functions
	// hidden output
	out << (int)hiddenActiv->GetType() << ' ' << (int)outputActiv->GetType() << '\n';

	// biases and weights
	// l0 weights ...
	// l0 biases ...
	// l1 weights ...
	// l1 biases ...

	bool f = true;
	for (const Layer& layer : layers)
	{
		if (f)
		{
			f = false; continue;
		}
		for (const double& w : layer.GetWeights())
		{
			out << w << ' ';
		}
		out << '\n';
		for (const double& b : layer.GetBiases())
		{
			out << b << ' ';
		}
		out << '\n';
	}
	out.close();
}

void net::Network::Load(std::string path)
{
	std::ifstream in{ path };
	in >> n_layers;

	std::vector<size_t> layer_c;
	for (size_t i = 0; i < n_layers; i++)
	{
		size_t c = 0;
		in >> c;
		layer_c.push_back(c);
	}

	this->layer_c = layer_c;

	int hiddenActiv = 0;
	int outputActiv = 0;

	in >> hiddenActiv;
	in >> outputActiv;

	this->hiddenActiv = std::move(std::make_unique<actf::Sigmoid>());
	this->outputActiv = std::move(std::make_unique<actf::Sigmoid>());

	layers.reserve(layer_c.size());

	layers.emplace_back(layer_c[0]);
	for (auto c = layer_c.begin() + 1; c != layer_c.end() - 1; ++c)
	{
		size_t i = c - layer_c.begin();
		layers.emplace_back(layers[i - 1], math::DMatrix{ 1, *c, 0.0 }, *c);
	}
	layers.emplace_back(layers[layers.size() - 1], math::DMatrix{ 1, layer_c[layer_c.size() - 1], 0.0 }, layer_c[layer_c.size() - 1]);

	for (Layer& layer : layers)
	{
		math::DMatrix weights{layer.GetWeights().GetRows(), layer.GetWeights().GetColumns()};
		for (size_t i = 0; i < layer.GetWeights().GetSize(); i++)
		{
			double w = 0.0;
			in >> w;
			weights[i] = w;
		}
		layer.SetWeights(weights);

		math::DMatrix biases{ layer.GetBiases().GetRows(), layer.GetBiases().GetColumns() };
		for (size_t i = 0; i < layer.GetBiases().GetSize(); i++)
		{
			double b = 0.0;
			in >> b;
			biases[i] = b;
		}
		layer.SetBiases(biases);
	}

	weight_grad.resize(layers.size());
	bias_grad.resize(layers.size());

	ClearGradients();

	in.close();
}

void net::Network::ApplyGradients(double learnRate)
{
	for (size_t i = 0; i < n_layers; ++i)
	{
		layers[i].SetWeights(layers[i].GetWeights() - weight_grad[i] * learnRate);
		layers[i].SetBiases(layers[i].GetBiases() - bias_grad[i] * learnRate);
	}
}

void net::Network::ClearGradients()
{
	for (auto grad = weight_grad.begin(); grad != weight_grad.end(); ++grad)
	{
		size_t i = grad - weight_grad.begin();
		*grad = math::DMatrix{ layers[i].GetWeights().GetRows(), layers[i].GetWeights().GetColumns() };
	}

	for (auto grad = bias_grad.begin(); grad != bias_grad.end(); ++grad)
	{
		size_t i = grad - bias_grad.begin();
		*grad = math::DMatrix{ layers[i].GetBiases().GetRows(), layers[i].GetBiases().GetColumns() };
	}
}

void net::Network::UpdateGradients(size_t layer_i, math::DMatrix nodeValues)
{
	weight_grad[layer_i] += layers[layer_i - 1].GetOutputs().GetTransposed() * nodeValues;
	bias_grad[layer_i] = nodeValues;
}

void net::Network::GetGradients(util::DataPoint<double>& dp)
{
	CalculateOutputs(dp);

	math::DMatrix nodeValues = OutputLayerValues(dp);
	UpdateGradients(n_layers - 1, nodeValues);

	for (size_t i = n_layers - 2; i > 0; --i)
	{
		nodeValues = HiddenLayerValues(i, nodeValues);
		UpdateGradients(i, nodeValues);
	}
}

math::DMatrix net::Network::OutputLayerValues(util::DataPoint<double>& dp) const
{
	return outputActiv->
		Derivative(layers[n_layers - 1].GetWeightedInputs())
		.Hadamard(cost->Derivative(dp));
}

math::DMatrix net::Network::HiddenLayerValues(size_t layer_i, math::DMatrix nodeValues) const
{
	return (nodeValues * layers[layer_i + 1].GetWeights().GetTransposed())
		.Hadamard(
			hiddenActiv
			->Derivative(layers[layer_i].GetWeightedInputs()));
}

math::DMatrix net::Network::Feed(math::DMatrix input)
{
	input = layers[0].Forward(input, *hiddenActiv, true); // the activation is not actually used
	for (auto layer = layers.begin() + 1; layer != layers.end() - 1; ++layer)
	{
		input = layer->Forward(input, *hiddenActiv);
	}
	input = layers[layers.size() - 1].Forward(input, *outputActiv);
	return input;
}

void net::Network::Learn(std::vector<util::DataPoint<double>>& batch, double learnRate)
{
	for (auto& dp : batch)
	{
		GetGradients(dp);
	}

	ApplyGradients(learnRate);
	ClearGradients();
}
