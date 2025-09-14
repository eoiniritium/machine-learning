#include <iostream>
#include <cmath>

#include "linear-algebra/matrix.hpp"
#include "machine-learning/neural-network.hpp"
#include "machine-learning/read-write-data.hpp"

const double alpha = 0.02;
const double learningRate = 0.3;
const size_t epochs = 5e4;
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

void train() {
    MachineLearning::NeuralNetwork net(
        {2, 5, 2, 1},
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

    MachineLearning::writeModel(net, "model.txt");
}

void loadFromFile() {
    MachineLearning::NeuralNetwork net(
        MachineLearning::loadModel("model.txt"),
        leakyRELU,
        leakyRELUPrime,
        costPrime
    );

    LinearAlgebra::Matrix Input(2, 1);

    Input.at(0, 0) = 0;
    Input.at(1, 0) = 0;

    std::cout << net.predict(Input).string() << std::endl;
}

int main() {
    //train();
    loadFromFile();

    return 0;
}