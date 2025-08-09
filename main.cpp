#include <iostream>

#include "linear-algebra/vector.hpp"
#include "linear-algebra/matrix.hpp"

int main() {
    
    LinearAlgebra::Matrix A(2, 2);
    LinearAlgebra::Matrix B(2, 2);

    A.at(0, 0) = 1;
    A.at(1, 0) = 3;
    A.at(0, 1) = 2;
    A.at(1, 1) = 4;

    B.at(0, 0) = 5;
    B.at(1, 0) = 7;
    B.at(0, 1) = 6;
    B.at(1, 1) = 8;


    std::cout << "A:\n" << A.string() << std::endl;
    std::cout << "\nB:\n" << B.string() << std::endl;


    std::cout << "\nA*B:\n" << (A*B).string() << std::endl;
    return 0;
}