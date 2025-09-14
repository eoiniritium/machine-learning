#pragma once

#include "linear-algebra/matrix.hpp"

#include <random>
#include <cmath>

namespace MachineLearning {
    class Layer {
        public:
        LinearAlgebra::Matrix w; // Weight to this layer
        LinearAlgebra::Matrix b; // Biases
        LinearAlgebra::Matrix z;
        LinearAlgebra::Matrix a;
        
        Layer(
            const size_t prevDimension,
            const size_t dimension
        ) {
            this->w = LinearAlgebra::Matrix(dimension, prevDimension, 0.0);
            this->b = LinearAlgebra::Matrix(dimension, 1, 0.0);
            this->a = LinearAlgebra::Matrix(dimension, 1, 0.0);
            this->z = LinearAlgebra::Matrix(dimension, 1, 0.0);

            // He normal initialisation
            std::random_device rd{};
            std::mt19937 gen(rd());

            for(size_t row = 0; row < dimension; ++row) {
                for(size_t column = 0; column < prevDimension; ++column) {
                   std::normal_distribution<double> n(0, sqrt(2.0/prevDimension));

                   this->w.at(row, column) = n(gen);
                }
                
            }
            
        }

        size_t dimension() const {
            return a.rows();
        }
    };
}