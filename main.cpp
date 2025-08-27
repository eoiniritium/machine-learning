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
        100,
        0.5
    );


    auto input = LinearAlgebra::Matrix({
        {2, 2},
        {1, 2}
    });

    std::cout << input.string() << std::endl;

    auto newMrix = testFunc(input);

    newMrix.at(0, 0) = 100;

    std::cout << input.string() << std::endl;

    //auto prediction = net.feedForward(input);

    

    return 0;
}