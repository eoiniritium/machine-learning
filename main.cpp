#include <iostream>
#include <vector>
#include <cmath>


#include "maths/matrix.hpp"
#include "maths/layer.hpp"
#include "maths/network.hpp"

const double alphaMax = 0.5;     // starting learning rate
const double alphaMin = 0.0001;   // floor learning rate
const size_t totalEpochs = 100000; // total epochs in the schedule

double sigmoid(const double x) { return 1.f/(1.f+exp(-x)); }
double Dsigmoid(const double x) { return sigmoid(x) * (1-sigmoid(x)); }
double cosineAnnealing(size_t epoch) {
    return alphaMin + 0.5 * (alphaMax - alphaMin) * (1 + std::cos(3.14159 * static_cast<double>(epoch) / totalEpochs));
}

int main() {

    Maths::MachineLearning::NeuralNetwork network;


    network
    .addLayer(Maths::MachineLearning::Layer(
        2, 5, sigmoid, Dsigmoid
    ), "input-layer")
    .addLayer(Maths::MachineLearning::Layer(
        5, 5, sigmoid, Dsigmoid
    ), "hidden-layer")
    .addLayer(Maths::MachineLearning::Layer(
        5, 1, sigmoid, Dsigmoid
    ), "output-layer");

    

    network["input-layer"].initKaimingNormal();
    network["hidden-layer"].initKaimingNormal();
    network["output-layer"].initKaimingNormal();

    

    network.setOrder({"input-layer", "hidden-layer", "output-layer"});

    auto trainingData = Maths::MachineLearning::generateXORTrainingData();

    network.learn(trainingData, totalEpochs, cosineAnnealing, 1);

    std::cout << network.feedForward(Maths::Matrix<double>({1, 0}, 2, 1)) << std::endl; // 1
    std::cout << network.feedForward(Maths::Matrix<double>({0, 1}, 2, 1)) << std::endl; // 1
    std::cout << network.feedForward(Maths::Matrix<double>({0, 0}, 2, 1)) << std::endl; // 0
    std::cout << network.feedForward(Maths::Matrix<double>({1, 1}, 2, 1)) << std::endl; // 0

    return 0;
}