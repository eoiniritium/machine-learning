#pragma once

#include <vector>
#include "linear-algebra/matrix.hpp"

namespace MachineLearning {
    typedef std::vector<std::pair<LinearAlgebra::Matrix, LinearAlgebra::Matrix>> TrainingData;
}