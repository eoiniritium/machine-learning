#pragma once

#include "defs.hpp"

#include "linear-algebra/matrix.hpp"


#include <iostream>
#include <cmath>


namespace MachineLearning {
    class Layer {
        public:
        LinearAlgebra::Matrix z, a;
        LinearAlgebra::Matrix biases;
        LinearAlgebra::Matrix weightsToNext;

        size_t dimension;

        Layer(const size_t dimension, const size_t nextDimension=0) {
            this->dimension = dimension;
            this->biases    = LinearAlgebra::Matrix(dimension, 1);
            //this->a         = LinearAlgebra::Matrix(dimension, 1);
            //this->z         = LinearAlgebra::Matrix(dimension, 1);

            if(nextDimension != 0) {
                this->weightsToNext = LinearAlgebra::Matrix(dimension, nextDimension);
            }
        }
    };

    class NeuralNetwork {
        private:
        std::vector<Layer> layers;

        public:
        NeuralNetwork(const std::vector<size_t> &sizes) {
            auto sizesCopy = sizes;
            sizesCopy.push_back(0);
            for(size_t l = 0; l < sizesCopy.size()-1; ++l) {
                Layer layer(sizesCopy[l], sizesCopy[l+1]);
            }
        }

        LinearAlgebra::Matrix feedForward(const LinearAlgebra::Matrix &Input) {
            std::cout << "Here 1" << std::endl;
            layers[0].z = LinearAlgebra::Matrix(Input);
            std::cout << "Here 2" << std::endl;
            layers[0].a = Input.vectorise(sigmoid);


            for(size_t i = 1; i < layers.size(); ++i) {
                layers[i].z = (layers[i-1].weightsToNext * layers[i-1].a) + layers[i].biases;
                layers[i].a = layers[i].z.vectorise(sigmoid);
            }

            return layers[layers.size() - 1].a; // Return activation of outputlayer
        }

        void train(
            const std::vector<std::pair<LinearAlgebra::Matrix, LinearAlgebra::Matrix>> &trainingData,
            const size_t epochs,
            const double learningRate
        ) {
            for(size_t epoch = 0; epoch < epochs; ++epoch) {
                for(size_t i = 0; i < trainingData.size(); ++i) {
                    this->feedForward(trainingData[i].first);
                    this->backPropagation(trainingData[i].second, learningRate);
                }
            }
        }


        private:

        void backPropagation(const LinearAlgebra:: Matrix &expected, const double learningRate) {
            size_t L = layers.size() - 1; // Index of output layer
            std::vector<LinearAlgebra::Matrix> deltas(L+1);

            // delta in output layer
            deltas[L] = (layers[L].a - expected).hadamardProduct(layers[L].z.vectorise(derivativeSigmoid));

            for(size_t i = 1; i <= L; ++i) {
                auto w_T = layers[L-i+1].weightsToNext.transpose();

                deltas[L-i] = (w_T*deltas[L-i+1]).hadamardProduct(layers[L-i].z.vectorise(derivativeSigmoid));
            }

            // Tweak Biases
            for(size_t l = 0; l < layers.size(); ++l) {
                layers[l].biases = layers[l].biases - learningRate*deltas[l];
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