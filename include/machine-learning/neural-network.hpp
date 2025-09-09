#pragma once

#include "defs.hpp"

#include "linear-algebra/matrix.hpp"
#include "utils.hpp"


#include <iostream>
#include <format>

#include <functional>
#include <algorithm>
#include <random>


namespace MachineLearning {

    class Layer {
        public:
        LinearAlgebra::Matrix w; // Weight to this layer
        LinearAlgebra::Matrix b; // Biases
        LinearAlgebra::Matrix z;
        LinearAlgebra::Matrix a;
        
        Layer(
            const size_t prevDimension,
            const size_t dimension,
            const std::pair<double, double> randomRange
        ) {
            this->w = LinearAlgebra::Matrix(dimension, prevDimension, randomRange);
            this->b = LinearAlgebra::Matrix(dimension, 1, randomRange);
            this->a = LinearAlgebra::Matrix(dimension, 1, 0);
            this->z = LinearAlgebra::Matrix(dimension, 1, 0);
        }

        size_t dimension() const {
            return a.columns();
        }
    };

    class NeuralNetwork {
        private:
        CostDerivativeFunc costPrime;
        ActivationFunc sigma, sigmaPrime;

        std::vector<Layer*> layers;
        
        public:

        NeuralNetwork(
            const std::vector<size_t> &dimensions,
            const std::pair<double, double> startRandomRange,
            const CostDerivativeFunc costPrime,
            const ActivationFunc sigma,
            const ActivationFunc sigmaPrime
        ) {
            /*
                Neural Network Constructor

                Arguments
                    dimension        - dimension of each layer.
                    startRandomRange - the range of random numbers the weights and biases should be initialised with
                    costPrime        - (Matrix expected, Matrix predicted) -> Matrix. Derivative of the cost function
            */
            if(dimensions.size() == 0) { throw std::invalid_argument("NeuralNetwork: Must have atleast 1 layer"); }
            
            this->costPrime = costPrime;
            this->sigma = sigma;
            this->sigmaPrime = sigmaPrime;

            
            // Input Layer. Has no previous layer
            layers.push_back(new Layer(
                0,
                dimensions[0],
                startRandomRange
            ));

            for(size_t l = 1; l < dimensions.size(); ++l) {
                layers.push_back(new Layer(
                    dimensions[l-1],
                    dimensions[l],
                    startRandomRange
                ));
            }
        }

        ~NeuralNetwork() {
            // TODO - No memory leaks please!
        }

        LinearAlgebra::Matrix predict(const LinearAlgebra::Matrix & Input) {
            this->feedForward(Input);

            return this->layers.back()->a;
        }

        void train(
            const TrainingData &trainingData,
            const size_t batchSize,
            const size_t epochs,
            const double learningRate,
            const size_t outputFrequency = 0
        ) {
            auto rng = std::default_random_engine {};
            for(size_t epoch = 0; epoch < epochs; ++epoch) {
                // Shuffle data
                auto data = trainingData;
                std::shuffle(std::begin(data), std::end(data), rng);

                while(!data.empty()) {
                    std::vector<std::vector<LinearAlgebra::Matrix>> batch;
                    for(size_t i = 0; i < batchSize && !data.empty(); ++i) {
                        auto pair = data.back();
                        
                        this->feedForward(pair.first);
                        batch.push_back(this->backPropagate(pair.second));
                        
                        data.pop_back();
                    }

                    this->updateBatches(batch, learningRate);
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

        std::vector<LinearAlgebra::Matrix> backPropagate(const LinearAlgebra::Matrix &expected) const {
            std::vector<LinearAlgebra::Matrix> deltas(this->layers.size()-1);

            deltas[deltas.size() - 1] = this->costPrime(expected, layers.back()->a).hadamardProduct(layers.back()->z);

            for(size_t i = deltas.size()-1; i--;) {
                size_t l = i + 1; // I think

                auto LHS = (layers[l+1]->w.transpose() * deltas[i+1]);
                auto RHS = layers[l]->z.vectorise(this->sigmaPrime);
                
                deltas[i] = LHS.hadamardProduct(RHS);
            }

            return deltas;
        }

        void updateBatches(const std::vector<std::vector<LinearAlgebra::Matrix>> &deltasVec, const double learningRate) {
            double m = deltasVec.size();

            for(size_t l = 1; l < layers.size(); ++l) {
                
                LinearAlgebra::Matrix summandWeights(layers[l]->w.rows(), layers[l]->w.columns(), 0.0);
                LinearAlgebra::Matrix summandBiases(layers[l]->b.rows(), layers[l]->b.columns(), 0.0);

                size_t deltas_i = l - 1; // Reindex from layers

                for(const auto & deltas : deltasVec) {
                    summandWeights = summandWeights + deltas[deltas_i] * layers[l-1]->a.transpose();
                    summandBiases = summandBiases + deltas[deltas_i];
                }

                layers[l]->w = layers[l]->w - (learningRate/m) * summandWeights;
                layers[l]->b = layers[l]->b - (learningRate/m) * summandBiases;
            }
        }
    };
}