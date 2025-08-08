#include <iostream>

#include "linear-algebra/vector.hpp"
#include "linear-algebra/matrix.hpp"

int main() {
    LinearAlgebra::Vector vec1({1, 2, 3, 4});
    LinearAlgebra::Vector vec2({1, 2, 3, 4});

    std::cout << vec1.dot(vec2) << std::endl;

    return 0;
}