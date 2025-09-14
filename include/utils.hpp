#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iterator>


namespace utils {
    std::vector<std::string> split(const std::string& str, char delim) {
        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string item;

        while (std::getline(ss, item, delim)) {
            tokens.push_back(item);
        }

        return tokens;
    }
}