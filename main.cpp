#include <iostream>
#include <cmath>

#include "linear-algebra/matrix.hpp"
#include "machine-learning/neural-network.hpp"
#include "machine-learning/load-data.hpp"

LinearAlgebra::Matrix costFunction (const LinearAlgebra::Matrix &Expected, const LinearAlgebra::Matrix &Predicted) {
    return Expected - Predicted;
}

double sigmoid(const double x) {
    return 1.0/(1.0+exp(x));
}

double sigmoidPrime(const double x) {
    return x * (1.0 - x);
}

int main() {
    MachineLearning::NeuralNetwork net(
        {2, 2, 1},
        std::make_pair(0.0, 0.1),
        costFunction,
        sigmoid,
        sigmoidPrime
    );

    auto trainingData = MachineLearning::loadTrainingData("train.txt");
    net.train(
        trainingData,
        2,
        1e4,
        0.5
    );

    LinearAlgebra::Matrix Input(2, 1);

    Input.at(0, 0) = 1;
    Input.at(1, 0) = 1;

    auto predction = net.predict(Input);

    std::cout << predction.string() << std::endl;


    return 0;
}