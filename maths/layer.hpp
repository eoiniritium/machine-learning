#pragma once

#include <vector>
#include <stdexcept>
#include <functional>
#include <random>
#include <tuple>
#include <chrono>

#include "matrix.hpp"

#include <iostream>

namespace Maths::MachineLearning {
    typedef std::function<double (double)> activationFunction;

    class Layer {
        private:
        Matrix<double> w, b;
        std::default_random_engine generator;
        activationFunction activationFn, DactivationFn;

        public:
        Layer(
            const size_t inputDimension, const size_t outputDimension,
            activationFunction activation, activationFunction Dactivation
        ) {
            w = Matrix<double>(outputDimension, inputDimension);
            b = Matrix<double>(outputDimension, 1);

            this->activationFn = activation;
            this->DactivationFn = Dactivation;

            generator.seed(std::random_device{}() ^ static_cast<unsigned>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count()
            ));
        }

        void initKaimingNormal(double gain = 2.f, bool fanIn = true) {
            auto n = w.shape().second;
            if(!fanIn) { n = w.shape().first; }

            std::normal_distribution<double> distribution(0.f, sqrt(gain/static_cast<double>(n)));


            w = w.applyElementWise([&](double _) {
                return distribution(generator);
            });
        }

        Matrix<double> preActivation(const Matrix<double> &inputData) {
            Matrix<double> linear;
            auto inputShapeCols = inputData.shape().second;

            if(inputShapeCols == 1) {
                linear = w*inputData + b;
            } else {
                linear = w*inputData + b.repeatColumn(inputShapeCols);
            }

            return linear;
        }

        // Call after preActivation
        Matrix<double> activation(const Matrix<double> &preActivation) const {
            return preActivation.applyElementWise(activationFn);
        }

        Matrix<double> Dactivation(const Matrix<double> &preActivation) const {
            return preActivation.applyElementWise(DactivationFn);
        }


        std::pair<size_t, size_t> shape() const {
            auto ws = w.shape();
            return std::pair<size_t, size_t>(ws.second, ws.first);
        }

        const Matrix<double> &weights() const { return w; }
        const Matrix<double> &biases()  const { return b; }

        void updateWeights(const Matrix<double> &delta) { w -= delta; }
        void updateBiases (const Matrix<double> &delta) { b -= delta; }
    };
}