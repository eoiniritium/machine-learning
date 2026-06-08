#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <unordered_map>
#include <format>
#include <iostream>

#include "matrix.hpp"
#include "layer.hpp"

namespace Maths::MachineLearning {
    typedef std::vector<std::pair<Matrix<double>, Matrix<double>>> TrainingData;
    typedef std::function<double (Matrix<double>, Matrix<double>)> LossFunction;
    typedef std::function<double (size_t)> LearningRateFunction;

    TrainingData generateXORTrainingData() {
        return {
            { Matrix<double>({0, 0}, 2, 1), Matrix<double>({0}, 1, 1) },
            { Matrix<double>({0, 1}, 2, 1), Matrix<double>({1}, 1, 1) },
            { Matrix<double>({1, 0}, 2, 1), Matrix<double>({1}, 1, 1) },
            { Matrix<double>({1, 1}, 2, 1), Matrix<double>({0}, 1, 1) }
        };
    }

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
            const TrainingData &trainingData,
            size_t epochs,
            LearningRateFunction learningRateFn,
            size_t batchSize,
            size_t printEpochStep = 100
        ) {
            std::vector<Matrix<double>> z(layers.size());
            std::vector<Matrix<double>> x(layers.size()+1);
            std::vector<Matrix<double>> deltas(layers.size());

            size_t n = trainingData[0].second.shape().first; // number of rows in expected output
            size_t L = layers.size();

            double runningError = 0.f;

            for(size_t epoch = 0; epoch < epochs; ++epoch) {

                runningError = 0.f;

                for(size_t observation = 0; observation < trainingData.size(); ++observation) {
                    x[0] = trainingData[observation].first;

                    // Forwards Pass
                    for(size_t l = 0; l < L; ++l) {
                        Layer *layer = &layers.at(layerOrder[l]);

                        z[l] = layer->preActivation(x[l]);
                        x[l+1] = layer->activation(z[l]);
                    }

                    auto errorMatrix = x[L] - trainingData[observation].second;

                    // Backwards Pass
                    deltas[L-1] = (2/n) * (layers.at(layerOrder[L-1]).Dactivation(z[L-1])).hadamard(errorMatrix);
                    for(int l = L-2; l >= 0; --l) {
                        Layer *layer     = &layers.at(layerOrder[l]);
                        Layer *nextLayer = &layers.at(layerOrder[l+1]);

                        deltas[l] = layer->Dactivation(z[l]).hadamard(nextLayer->weights().transpose() * deltas[l+1]);
                    }

                    // Gradients update
                    for(size_t l = 0; l < L; ++l) {
                        Layer *layer = &layers.at(layerOrder[l]);

                        layer->updateWeights(learningRateFn(epoch) * deltas[l] * x[l].transpose());
                        layer->updateBiases (learningRateFn(epoch) * deltas[l]);
                    }

                    // Telematics
                    if(epoch % printEpochStep == 0) {
                        runningError += (1/n) * (errorMatrix * errorMatrix.transpose())[0, 0];
                    }
                }

                // Telematics
                if(epoch % printEpochStep == 0) {
                    std::cout << std::format(
                        "Epoch {}: MSE = {}",
                        epoch, runningError/static_cast<double>(trainingData.size())
                    ) << std::endl;
                }
            }
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