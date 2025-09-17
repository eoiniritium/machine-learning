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


int myMain(std::string modelWeights, std::string inputVector) {
    /* Jank way to get it to work with WASM
    

        argv[0]: app name
        argv[1]: modelWeights pulled directly from an outputted model
        argv[2]: input vector, delimed by commas
    */
    

    // --- Load In Model ---
    std::vector<std::string> lines = utils::split(modelWeights, '\n');

    MachineLearning::NetworkParameters layers;

    std::vector<std::vector<double>> weights, biases;

    MachineLearning::LayerParameters firstLayer;

    for(size_t i = 0; i < lines.size(); ++i) {
        std::string line = lines[i];

        if(line == "") {
            MachineLearning::LayerParameters layer;
            layer.weights = LinearAlgebra::Matrix(weights);
            layer.biases  = LinearAlgebra::Matrix(biases );

            std::cout << layer.weights.string() << std::endl << std::endl;

            layers.push_back(layer);

            weights.clear();
            biases.clear();
            continue;
        }

        auto split = utils::split(line, ' ');

        auto weightsRowString = utils::split(split[0], ',');
        auto biasesRowString  = utils::split(split[1], ',');

        std::vector<double> weightsRow(weightsRowString.size());
        std::vector<double> biasesRow(biasesRowString.size());

        std::transform(
            weightsRowString.begin(), weightsRowString.end(),
            weightsRow.begin()      ,
            [](std::string const& val) {return std::stod(val);}
        );
        std::transform(
            biasesRowString.begin(), biasesRowString.end(),
            biasesRow.begin()      ,
            [](std::string const& val) {return std::stod(val);}
        );

        weights.push_back(weightsRow);
        biases.push_back(biasesRow);
    }

    // --- Initalise Model ---
    MachineLearning::NeuralNetwork net(
        layers,
        leakyRELU,
        leakyRELUPrime,
        costPrime
    );

    // --- Predict ---
    auto splitInput = utils::split(inputVector, ',');
    
    LinearAlgebra::Matrix input(splitInput.size(), 1);

    for(size_t i = 0; i < splitInput.size(); ++i) {
        input.at(i, 0) = std::stod(splitInput[i]);
    }
    
    auto output = net.predict(input).transpose();

    std::cout << utils::join<double>(" ", output.get2DVector()[0]);

    return 0;
}

int main() {
    std::ifstream t("model.txt");
    std::string str((std::istreambuf_iterator<char>(t)),
    std::istreambuf_iterator<char>());

    myMain(str, "0,0");
}
/*
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(ml) {
    emscripten::function("myMain", &myMain, emscripten::allow_raw_pointers());
}
*/