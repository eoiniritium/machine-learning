#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iterator>
#include <format>


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

    template <typename T>
    std::string join(const std::string &joinChar, const std::vector<T> &vec) {
        std::string ret = std::to_string(vec[0]);

        for(size_t i = 1; i < vec.size(); ++i) {
            ret += std::format("{}{}", joinChar, std::to_string(vec[i]));
        }

        return ret;
    }
}