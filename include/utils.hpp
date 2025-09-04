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

    class Logger {
        private:
        std::string path;
        std::vector<std::string> lines;

        public:
        Logger() {};
        Logger(const std::string path) {
            this->path = path;

            this->lines = std::vector<std::string>();
        }

        void addLine(const std::string line) {
            lines.push_back(line);
        }

        void write() {
            std::ofstream outputFile(this->path);
            std::ostream_iterator<std::string> outputIterator(outputFile, "\n");
            std::copy(std::begin(this->lines), std::end(this->lines), outputIterator);
        }

        ~Logger() {
            this->write();
        }
    };
}