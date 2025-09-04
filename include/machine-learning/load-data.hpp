#pragma once

#include "machine-learning/defs.hpp"
#include "utils.hpp"

#include "linear-algebra/matrix.hpp"

#include <vector>
#include <string>
#include <fstream>

#include <iostream>


namespace MachineLearning {
    TrainingData loadTrainingData(const std::string &path) {
        TrainingData ret = {};
        std::ifstream file(path);
        std::string line;


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
}