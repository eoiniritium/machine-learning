#pragma once

#include "matrix.hpp"

#include <cmath>


namespace MachineLearning {
    class NeuralNetwork {
        private:
        size_t inputSize, hiddenSize, outputSize;
        LinearAlgebra::Matrix input2hiddenWeights, hidden2outputWeights;
        LinearAlgebra::Matrix hiddenBias, outputBias;


        public:
        NeuralNetwork(const size_t inputSize, const size_t hiddenSize, const size_t outputSize) {
            this->inputSize  = inputSize;
            this->hiddenSize = hiddenSize;
            this->outputSize = outputSize;

            this->input2hiddenWeights  = LinearAlgebra::Matrix(this->inputSize, this->hiddenSize);
            this->hidden2outputWeights = LinearAlgebra::Matrix(this->hiddenSize, this->outputSize);
        
            this->hiddenBias = LinearAlgebra::Matrix(1, this->hiddenSize);
            this->outputBias = LinearAlgebra::Matrix(1, this->outputSize);
        }

        LinearAlgebra::Matrix feedFoward(const LinearAlgebra::Matrix input) const {            
            
            auto hiddenActivation = (input * this->input2hiddenWeights).apply(this->sigmoid);
            auto outputActivation = (hiddenActivation * hidden2outputWeights).apply(this->sigmoid);

            return outputActivation;
        }

        private:

        static double sigmoid(const double x) { return 1/(1+ exp(-x)); }
        static double sigmoidDerivative(const double x) {return x * (1 - x); }
    };
}