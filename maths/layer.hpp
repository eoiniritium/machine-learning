#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <random>

#include "matrix.hpp"

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


            applyInit([&](){
                return distribution(generator);
            });
        }

        Matrix<double> feedForward(const Matrix<double> &inputData) {
            Matrix<double> linear;
            auto inputShape = inputData.shape();
            if(inputShape.second == 1) {
                linear = weights*inputData + weights;
            } else {
                linear = weights*inputData + weights.repeatColumn(inputShape.second);
            }

            return activation(linear);
        }

        private:
        void applyInit(const std::function<double (void)> &elementWiseInitialisation) {
            auto shape = weights.shape();

            for(size_t i = 0; i < shape.first; ++i) {
                for(size_t j = 0; j < shape.second; ++j) {
                    weights[i, j] = elementWiseInitialisation();
                }
            }
        }
    };
}