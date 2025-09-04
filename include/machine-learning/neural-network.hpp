#pragma once

#include "defs.hpp"

#include "linear-algebra/matrix.hpp"
#include "utils.hpp"


#include <iostream>
#include <format>
#include <cmath>


namespace MachineLearning {
    class Layer {
        public:
        LinearAlgebra::Matrix z, a, biases, weights2next;
        size_t dimension;

        Layer(const size_t dimension, const size_t nextLayerDimension) {
            this->dimension = dimension;

            this->z      = LinearAlgebra::Matrix(dimension, 1);
            this->a      = LinearAlgebra::Matrix(dimension, 1);
            this->biases = LinearAlgebra::Matrix(dimension, 1, true);

            if (nextLayerDimension != 0) {
                this->weights2next = LinearAlgebra::Matrix(nextLayerDimension, dimension, true);
            }
        }
    };

    class NeuralNetwork {
        private:
        std::vector<Layer*> layers;
        utils::Logger logger;

        public:
        NeuralNetwork(const std::vector<size_t> &dimensions) {
            this->layers = std::vector<Layer*>(dimensions.size());

            auto temp = dimensions;
            temp.push_back(0);

            for(size_t l = 0; l < layers.size(); ++l) {
                layers[l] = new Layer(temp[l], temp[l+1]);
            }

            this->logger = utils::Logger("log.txt");
        }

        LinearAlgebra::Matrix predict(const LinearAlgebra::Matrix &Input) {
            return this->feedForward(Input);
        }

        void train(
            const TrainingData &trainingData,
            const size_t epochs,
            const double learningRate,
            const size_t outputFrequency = 0
        ) {
            for(size_t epoch = 0; epoch < epochs; ++epoch) {
                LinearAlgebra::Matrix error(trainingData[0].second.rows(), 1);

                for(size_t i = 0; i < trainingData.size(); ++i) {
                    auto prediction = this->feedForward(trainingData[i].first);
                    this->backPropagation(trainingData[i].second, learningRate);

                    if(outputFrequency) {
                        error = error + (prediction - trainingData[i].second).vectorise([](double x) {return std::abs(x);});
                    }
                } 

                if(outputFrequency && epoch % outputFrequency == 0) {
                    double total = error.sumOverColumn(0);

                    std::cout << std::format("Epoch: {} Total error: {}", epoch, total) << std::endl;
                }
            }
        }


        private:
        LinearAlgebra::Matrix feedForward(const LinearAlgebra::Matrix &Input) {

            layers[0]->z = LinearAlgebra::Matrix(Input);
            layers[0]->a = Input.vectorise(sigmoid);


            for(size_t i = 1; i < layers.size(); ++i) {
                layers[i]->z = (layers[i-1]->weights2next * layers[i-1]->a) + layers[i]->biases;
                layers[i]->a = layers[i]->z.vectorise(sigmoid);
            }

            return layers[layers.size() - 1]->a; // Return activation of outputlayer
        }

        void backPropagation(const LinearAlgebra:: Matrix &expected, const double learningRate) {
            size_t L = layers.size() - 1; // Index of output layer
            std::vector<LinearAlgebra::Matrix> deltas(L+1);

            // delta in output layer
            deltas[L] = (layers[L]->a - expected).hadamardProduct(layers[L]->z.vectorise(derivativeSigmoid));

            for(size_t l = L; l--; ) {
                auto w_T = layers[l]->weights2next.transpose();
                auto sigmaPrime = layers[l]->z.vectorise(derivativeSigmoid);

                deltas[l] = (w_T * deltas[l+1]).hadamardProduct(sigmaPrime);
            }

            // Tweak Biases
            for(size_t l = 0; l < layers.size(); ++l) {
                auto m = layers[l]->dimension;
                
                layers[l]->biases = layers[l]->biases - learningRate * deltas[l];
            }
            
            // Tweak Weights
            for(size_t l = 0; l < layers.size()-1; ++l) {
                layers[l]->weights2next = layers[l]->weights2next - learningRate * deltas[l+1] * layers[l]->a.transpose();
            }
        }

        static double sigmoid(double x) {
            return 1/(1+exp(x));
        }
        static double derivativeSigmoid(double x) {
            return x * (1 - x);
        }
    };
}