#include <iostream>
#include "linear-algebra/matrix.hpp"
#include "machine-learning/neural-network.hpp"
#include "machine-learning/load-data.hpp"

int main() {
    MachineLearning::NeuralNetwork net({2, 3, 1});


    auto trainingData = MachineLearning::loadTrainingData("train.txt");

    net.train(
        trainingData,
        100,
        0.5
    );


    auto input = LinearAlgebra::Matrix(2, 1);
    input.at(0, 0) = 1;
    input.at(1, 0) = 0;

    auto prediction = net.feedForward(input);

    return 0;
}