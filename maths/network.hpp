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
        LossFunction lossFunction, DlossFunction;

        public:
        NeuralNetwork(const LossFunction lossFunction, const LossFunction DlossFunction) {
            this->lossFunction = lossFunction;
            this->DlossFunction = DlossFunction;
        }

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
                carry = layers.at(layerName).feedForward(carry);
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