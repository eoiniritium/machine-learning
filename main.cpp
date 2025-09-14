#include <iostream>
#include <cmath>

#include "linear-algebra/matrix.hpp"
#include "machine-learning/neural-network.hpp"
#include "machine-learning/load-data.hpp"

const double alpha = 0.02;
const double learningRate = 0.3;
const size_t epochs = 4e4;
const size_t batchSize = 1;
const size_t outputFrequency = 1000;

LinearAlgebra::Matrix costPrime (const LinearAlgebra::Matrix &Expected, const LinearAlgebra::Matrix &Predicted) {
    return Predicted - Expected;
}

double leakyRELU(const double x) {
    if (x > 0) return x;
    return alpha*x;
}

double leakyRELUPrime(const double x) {
    if (x > 0) return 1.0;
    return alpha;
}

int main() {
    MachineLearning::NeuralNetwork net(
        {2, 3, 2, 1},
        leakyRELU,
        leakyRELUPrime,
        costPrime
    );

    auto trainingData = MachineLearning::loadTrainingData("train.txt");
    net.train(
        trainingData,
        batchSize,
        epochs,
        learningRate,
        outputFrequency
    );

    LinearAlgebra::Matrix Input(2, 1);

    Input.at(0, 0) = 0;
    Input.at(1, 0) = 1;

    auto predction = net.predict(Input);

    std::cout << predction.string() << std::endl;


    return 0;
}