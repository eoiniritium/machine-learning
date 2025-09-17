#pragma once

#include "defs.hpp"
#include "layer.hpp"

#include "linear-algebra/matrix.hpp"

#include "utils.hpp"


#include <iostream>
#include <format>

#include <functional>
#include <algorithm>
#include <random>
#include <cmath>


namespace MachineLearning {

    class NeuralNetwork {
        private:
        std::vector<Layer*> layers;
        ActivationFunc sigma, sigmaPrime;
        CostDerivativeFunc costPrime;
        public:
        NeuralNetwork(
            const std::vector<size_t> &dimensions,
            ActivationFunc sigma, ActivationFunc sigmaPrime,
            CostDerivativeFunc costPrime
        ) {
            /*
                New Neural Network Constructor

                Arguments
                    dimension  - dimension of each layer.
                    sigma      - (double) -> double. Activation function
                    sigmaPrime - (double) -> double. Derivative of the activation function
                    costPrime  - (Matrix expected, Matrix predicted) -> Matrix. Derivative of the cost function
            */
            if(dimensions.size() == 0) { throw std::invalid_argument("NeuralNetwork: Must have atleast 1 layer"); }

            this->sigma = sigma;
            this->sigmaPrime = sigmaPrime;
            this->costPrime = costPrime;

            // Input Layer. Has no previous layer. (1, N) to make writing to file easier
            layers.push_back(new Layer(1, dimensions[0]));

            for(size_t l = 1; l < dimensions.size(); ++l) {
                layers.push_back(new Layer(dimensions[l-1], dimensions[l]));
            }
        }

        NeuralNetwork(
            const NetworkParameters &networkParameters,
            ActivationFunc sigma, ActivationFunc sigmaPrime,
            CostDerivativeFunc costPrime
        ) {
            /*
                Neural Network from existing weights and biases constructor

                Arguments
                    networkParameters - Network's weights and biases loaded from file via MachineLearning::loadModel
                    sigma             - function: (double) -> double. Activation function
                    sigmaPrime        - function: (double) -> double. Derivative of the activation function
                    costPrime         - function: (Matrix expected, Matrix predicted) -> Matrix. Derivative of the cost function
            */

            this->sigma = sigma;
            this->sigmaPrime = sigmaPrime;
            this->costPrime = costPrime;

            for(size_t l = 0; l < networkParameters.size(); ++l) {
                layers.push_back(new Layer(networkParameters[l]));
            }
        }

        ~NeuralNetwork() {
            for(size_t i = 0; i < this->layers.size(); ++i) {
                delete this->layers[i];
            }
        }

        NetworkParameters getParameters() const {
            NetworkParameters ret;
            for(size_t l = 0; l < this->layers.size(); ++l) {
                LayerParameters layer;
                layer.dimension = this->layers[l]->dimension();
                layer.weights = this->layers[l]->w;
                layer.biases  = this->layers[l]->b;

                ret.push_back(layer);
            }

            return ret;
        }


        LinearAlgebra::Matrix predict(const LinearAlgebra::Matrix & Input) {
            this->feedForward(Input);

            return this->layers.back()->a;
        }

        void train(
            const TrainingData trainingData,
            const size_t batchSize,
            const size_t epochs,
            const double learningRate,
            const size_t outputFrequency = 0
        ) {
            auto rng = std::default_random_engine {};

            const size_t N = trainingData.size();

            for(size_t epoch = 0; epoch < epochs; ++epoch) {
                // Shuffle data
                auto data = trainingData;
                std::shuffle(std::begin(data), std::end(data), rng);

                LinearAlgebra::Matrix epochError(layers.back()->dimension(), 1, 0.0);

                while(!data.empty()) {
                    std::vector<DeltasActivations> batch;
                    for(size_t i = 0; i < batchSize && !data.empty(); ++i) {
                        auto pair = data.back();

                        auto prediction = this->predict(pair.first);
                        batch.push_back(this->backPropagate(pair.second));

                        data.pop_back();

                        if(outputFrequency && epoch % outputFrequency == 0) {
                            epochError = epochError + (1.0/N)*(prediction - pair.second).vectorise([](const double x) {return fabs(x);});
                        }
                    }

                    this->updateBatches(batch, learningRate);
                }

                if (outputFrequency && epoch % outputFrequency == 0) {
                    std::cout << std::format("(Epoch: {}) Average error: {}", epoch, epochError.sumOverColumn(0)) << std::endl;
                }
            }
        }

        private:
        void feedForward(const LinearAlgebra::Matrix & Input) {
            layers[0]->a = Input;

            for(size_t l = 1; l < layers.size(); ++l) {
                layers[l]->z = layers[l]->w * layers[l-1]->a + layers[l]->b;
                layers[l]->a = layers[l]->z.vectorise(this->sigma);
            }
        }

        DeltasActivations backPropagate(const LinearAlgebra::Matrix &expected) const {
            std::vector<LinearAlgebra::Matrix> deltas(this->layers.size()-1);
            std::vector<LinearAlgebra::Matrix> activations(this->layers.size());


            // Save activations
            for(size_t l = 0; l < this->layers.size(); ++l) {
                activations[l] = this->layers[l]->a;
            }

            // Calculate Deltas
            auto LHS = this->costPrime(expected, layers.back()->a);
            auto RHS = layers.back()->z.vectorise(this->sigmaPrime);
            deltas[deltas.size() - 1] = LHS.hadamardProduct(RHS);

            for(size_t i = deltas.size()-1; i--;) {
                size_t l = i + 1;

                auto LHS = (layers[l+1]->w.transpose() * deltas[i+1]);
                auto RHS = layers[l]->z.vectorise(this->sigmaPrime);

                deltas[i] = LHS.hadamardProduct(RHS);
            }

            DeltasActivations ret;
            ret.deltas = deltas;
            ret.activations = activations;
            return ret;
        }

        void updateBatches(const std::vector<DeltasActivations> &deltasVec, const double learningRate) {
            double m = deltasVec.size();

            for(size_t l = 1; l < layers.size(); ++l) {
                size_t deltas_i = l - 1; // Reindex from layers

                LinearAlgebra::Matrix summandWeights(layers[l]->w.rows(), layers[l]->w.columns(), 0.0);
                LinearAlgebra::Matrix summandBiases(layers[l]->b.rows(), layers[l]->b.columns(), 0.0);

                for(const DeltasActivations &deltasActivation : deltasVec) {
                    auto deltas = deltasActivation.deltas;
                    auto activations =  deltasActivation.activations;
                    
                    summandWeights = summandWeights + deltas[deltas_i] * activations[l-1].transpose();
                    summandBiases = summandBiases + deltas[deltas_i];
                }

                layers[l]->w = layers[l]->w - (learningRate/m) * summandWeights;
                layers[l]->b = layers[l]->b - (learningRate/m) * summandBiases;
            }
        }

    };
}