#include<iostream>

#include "maths/matrix.hpp"

int main() {

    auto mtx1 = Maths::Matrix<double>({
        {1, 2},
        {3, 4},
    });
    auto mtx2 = Maths::Matrix<double>({
        {0, 2},
        {3, 0},
    });

    std::cout << 3*mtx2 << std::endl;
}