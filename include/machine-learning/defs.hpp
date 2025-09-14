#pragma once

#include <vector>
#include "linear-algebra/matrix.hpp"

namespace MachineLearning {
    struct DeltasActivations{
        std::vector<LinearAlgebra::Matrix> deltas;
        std::vector<LinearAlgebra::Matrix> activations;
    };

    typedef std::vector<std::pair<LinearAlgebra::Matrix, LinearAlgebra::Matrix>> TrainingData;
    typedef std::function<LinearAlgebra::Matrix (const LinearAlgebra::Matrix &, const LinearAlgebra::Matrix &)> CostDerivativeFunc;
    typedef std::function<double (const double)> ActivationFunc;
}