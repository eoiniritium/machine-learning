#pragma once

#include <vector>
#include <stdexcept>
#include <functional>
#include <random>
#include <tuple>

#include "matrix.hpp"

#include <iostream>

namespace Maths::MachineLearning {
    typedef std::function<double (double)> activationFunction;

    class Layer {
        private:
        Matrix<double> weights, bias;
        std::default_random_engine generator;
        activationFunction activation, Dactivation;

        public:
        Layer(
            const size_t inputDimension, const size_t outputDimension,
            activationFunction activation, activationFunction Dactivation
        ) {
            weights = Matrix<double>(outputDimension, inputDimension);
            bias = Matrix<double>(outputDimension, 1);

            this->activation = activation;
            this->Dactivation = Dactivation;
        }

        void initKaimingNormal(double gain = 2.f, bool fanIn = true) {
            auto n = weights.shape().second;
            if(!fanIn) { n = weights.shape().first; }

            std::normal_distribution<double> distribution(0.f, sqrt(gain/static_cast<double>(n)));


            weights = weights.applyElementWise([&](double _) {
                return distribution(generator);
            });
        }

        Matrix<double> feedForward(const Matrix<double> &inputData) {
            Matrix<double> linear;
            auto inputShapeCols = inputData.shape().second;

            if(inputShapeCols == 1) {
                linear = weights*inputData + bias;
            } else {
                linear = weights*inputData + bias.repeatColumn(inputShapeCols);
            }

            return linear.applyElementWise(activation);
        }
        Matrix<double> DfeedForward(const Matrix<double> &inputData) {
            Matrix<double> linear;
            auto inputShape = inputData.shape();
            if(inputShape.second == 1) {
                linear = weights*inputData + bias;
            } else {
                linear = weights*inputData + bias.repeatColumn(inputShape.second);
            }

            return linear.applyElementWise(Dactivation)*weights;
        }

        std::pair<size_t, size_t> shape() const {
            auto ws = weights.shape();
            return std::pair<size_t, size_t>(ws.second, ws.first);
        }
    };
}