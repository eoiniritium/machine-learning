#include <iostream>
#include "matrix.hpp"

int main() {

    LinearAlgebra::Matrix A({
        {1, 2},
        {3, 4}
    });

    LinearAlgebra::Matrix B({
        {5, 6},
        {7, 8}
    });
    
    
    std::cout << (A*B).string() << std::endl;

    return 0;
}