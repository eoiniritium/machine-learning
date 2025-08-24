#pragma once

#include "matrix.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <span>


namespace MachineLearning {
    class NeuralNetwork {
        private:
        typedef LinearAlgebra::Matrix     Matrix;
        typedef std::pair<Matrix, Matrix> TrainingPair;

        std::vector<Matrix> weights, biases;
    

        public:
        NeuralNetwork(const std::vector<size_t> &sizes) {
            
            for(size_t layer = 1; layer < sizes.size(); ++layer) {
                biases.push_back(LinearAlgebra::Matrix(sizes[layer], 1));
                weights.push_back(LinearAlgebra::Matrix(sizes[layer-1], sizes[layer]));
            }
        }

        LinearAlgebra::Matrix feedForward(const LinearAlgebra::Matrix &Input) const {
            auto a = Input;

            for(size_t i = 0; i < weights.size(); ++i) {
                a = ((weights[i] * a) + biases[i]).apply(sigmoid);
            }

            return a;
        }

        void train();


        private:

        void backPropagation() {

        }

        std::vector<std::vector<TrainingPair>> createBatches(const std::vector<TrainingPair> &trainingData, size_t batchSize) const {
            bool perfectPartition = (trainingData.size() % batchSize == 0);
            std::vector<std::vector<TrainingPair>> ret(trainingData.size()/batchSize + !perfectPartition);

            size_t batchCount = 0;
            for(size_t i = 0; i < trainingData.size(); ++i) {
                if(i == batchSize) { ++batchSize; }
                batchSize.pu;
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