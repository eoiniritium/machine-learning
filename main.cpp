#include <iostream>
#include <vector>

#include "maths/matrix.hpp"
#include "maths/layer.hpp"
#include "maths/network.hpp"

double sigmoid(const double x) { return 1.f/(1.f+exp(-x)); }
double Dsigmoid(const double x) { return sigmoid(x) * (1-sigmoid(x)); }

int main() {

    Maths::MachineLearning::NeuralNetwork network;


    network.addLayer(Maths::MachineLearning::Layer(
        2, 5, sigmoid, Dsigmoid
    ), "input-layer")
    .addLayer(Maths::MachineLearning::Layer(
        5, 1, sigmoid, Dsigmoid
    ), "output-layer");

    

    network["input-layer"].initKaimingNormal();
    network["output-layer"].initKaimingNormal();

    

    network.setOrder({"input-layer", "output-layer"});

    auto trainingData = Maths::MachineLearning::generateXORTrainingData();

    network.learn(trainingData, 1e8, 0.1, 1);

    return 0;
}