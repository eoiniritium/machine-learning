#include <iostream>
#include "linear-algebra/matrix.hpp"
#include "machine-learning/neural-network.hpp"
#include "machine-learning/load-data.hpp"

LinearAlgebra::Matrix testFunc(const LinearAlgebra::Matrix &Input) {
    const auto newMatrix = Input;

    return newMatrix;
}

int main() {
    MachineLearning::NeuralNetwork net({2, 3, 1});

    auto trainingData = MachineLearning::loadTrainingData("train.txt");

    net.train(
        trainingData,
        1e6,
        0.1,
        10000
    );



    LinearAlgebra::Matrix Input(2, 1);

    Input.at(0, 0) = 0;
    Input.at(1, 0) = 1;

    auto predction = net.predict(Input);

    std::cout << predction.string() << std::endl;

    return 0;
}