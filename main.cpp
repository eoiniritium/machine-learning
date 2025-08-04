#include <iostream>

#include "linear-algebra/vector.hpp"

int main() {
    linalg::Vector vec1({-1, 6});
    linalg::Vector vec2({1, 2});

    std::cout << vec1.scale(2).string();

    return 0;
}