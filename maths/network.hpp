#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <unordered_map>
#include <format>

#include "matrix.hpp"
#include "layer.hpp"

namespace Maths::MachineLearning {
    typedef std::vector<std::pair<Matrix<double>, Matrix<double>>> TraningData;
    typedef std::function<double (Matrix<double>, Matrix<double>)> LossFunction;

    class NeuralNetwork {
        private:
        std::unordered_map<std::string, Layer> layers;
        std::vector<std::string> layerOrder;

        public:
        NeuralNetwork &addLayer(const Layer &layer, std::string name) {
            auto result = layers.insert({name, layer});
            if(!result.second) { throw std::invalid_argument("Layer names must be unique!"); }

            return *this;
        }

        NeuralNetwork &setOrder(const std::vector<std::string> &order) {
            if(order.size() != layers.size()) {
                throw std::invalid_argument("Order must be equal in length to number of layers in network.");
            }
            for(auto name: order) {
                if(layers.find(name) == layers.end()) {
                    throw std::invalid_argument(std::format("Could not find layer {}", name));
                }
            }

            this->layerOrder = order;

            return *this;
        }

        Matrix<double> feedForward(const Matrix<double> &inputData) {
            auto carry = inputData;

            for(auto layerName: layerOrder) {
                Layer *layer = &layers.at(layerName);

                carry = layer->activation(layer->preActivation(carry));
            }

            return carry;
        }

        // Learn
        void learn(
            const TraningData &traningData,
            size_t epochs,
            double learningRate,
            size_t batchSize
        ) {
            std::vector<Matrix<double>> z(layers.size());
            std::vector<Matrix<double>> x(layers.size()+1);
            std::vector<Matrix<double>> deltas(layers.size());

            size_t n = traningData[0].second.shape().first; // number of rows in expected output
            size_t L = layers.size();

            for(size_t epoch = 0; epoch < epochs; ++epoch) {
            for(size_t observation = 0; observation < traningData.size(); ++observation) {
                x[0] = traningData[observation].first;

                // Forwards Pass
                for(size_t l = 0; l < L; ++l) {
                    Layer *layer = &layers.at(layerOrder[l-1]);

                    z[l] = layer->preActivation(x[l]);
                    x[l+1] = layer->activation(z[l]);
                }

                // Backwards Pass

                deltas[L-1] = (2/n) * (layers.at(layerOrder[L-1]).Dactivation(z[L-1]));
                for(size_t l = 1; l < layers.size() + 1; ++l) {
                    Layer *layer = &layers.at(layerOrder[l-1]);


                }

            }}

        }

        Layer operator[](std::string name) const {
            if(layers.find(name) == layers.end()) {
                throw std::invalid_argument(std::format("Could not find layer {}", name));
            }

            return layers.at(name);
        }
        Layer &operator[](std::string name) {
            if(layers.find(name) == layers.end()) {
                throw std::invalid_argument(std::format("Could not find layer {}", name));
            }

            return layers.at(name);
        }
    };
}