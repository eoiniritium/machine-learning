#pragma once

#include "machine-learning/defs.hpp"
#include "utils.hpp"

#include "linear-algebra/matrix.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#include <format>
#include <iostream>


namespace MachineLearning {
    TrainingData loadTrainingData(const std::string &path) {
        TrainingData ret = {};
        std::ifstream file(path);
        std::string line;


        size_t i = 1;
        while(std::getline(file, line)) {

            auto split = utils::split(line, ' ');

            auto lhs = utils::split(split[0], ',');
            auto rhs = utils::split(split[1], ',');

            LinearAlgebra::Matrix input(lhs.size(), 1);
            LinearAlgebra::Matrix expected(rhs.size(), 1);


            for(size_t i = 0; i < lhs.size(); ++i) {
                input.at(i, 0) = std::stod(lhs[i]);
            }

            for(size_t i = 0; i < rhs.size(); ++i) {
                expected.at(i, 0) = std::stod(rhs[i]);
            }

            ret.push_back(std::make_pair(input, expected));
        }

        return ret;
    }

    NetworkParameters loadModel(const std::string &path) {
        NetworkParameters layers;
        
        std::ifstream file(path);
        std::string line;

        std::vector<std::vector<double>> weights, biases;
        while(std::getline(file, line)) {
            if(line == "") {
                LayerParameters layer;
                layer.weights = LinearAlgebra::Matrix(weights);
                layer.biases  = LinearAlgebra::Matrix(biases );

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

        return layers;
    }

    void writeModel(const NeuralNetwork &model, const std::string path) {
        NetworkParameters params = model.getWeightsAndBiases();
        std::ofstream outfile(path);

        for(size_t l = 0; l < params.size(); ++l) {
            auto weights = params[l].weights.get2DVector();
            auto biases  = params[l].biases.get2DVector();

            for(size_t row = 0; row < weights.size(); ++row) {
                for(size_t col = 0; col < weights[row].size() - 1; ++col) {
                    outfile << weights[row][col] << ",";
                }
                outfile << weights[row].back() << " " << biases[row][0] << std::endl;
            }

            outfile << std::endl;
        }
    }
}