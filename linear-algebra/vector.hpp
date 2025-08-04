#pragma once

#include <vector>
#include <string>
#include <stdexcept>

namespace linalg {
    class Vector {
        public:    
        std::vector<double> vec;

        Vector(size_t n) {
            this->vec = std::vector<double>(n);
        }

        Vector(std::vector<double> vec) {
            this->vec = vec;
        }

        Vector(Vector &copy) {
            this->vec = copy.vec;
        }

        double operator   [](size_t i) const {return this->vec[i];}
        double & operator [](size_t i)       {return this->vec[i];}

        double dot(Vector &other) {
            if(this->size() != other.size()) {
                throw std::invalid_argument("Both vectors need to be the same size");
            }

            double sum = 0;

            for(size_t i = 0; i < this->size(); ++i) {
                sum += (*this)[i] * other[i];
            }

            return sum;
        }

        Vector scale(double scalar) {
            Vector ret(*this);

            for(size_t i = 0; i < ret.size(); ++i) {
                ret[i] *= scalar;
            }

            return ret;
        }

        Vector add(Vector &other) {
            Vector ret(other);

            for(size_t i = 0; i < ret.size(); ++i) {
                ret[i] += vec[i];
            }

            return ret;
        }

        size_t size() {
            return this->vec.size();
        }

        std::string string() {
            std::string ret = "";
            for(size_t i = 0; i < this->size(); ++i) {
                ret += std::to_string((*this)[i]) + " ";
            }

            ret.pop_back();

            return ret;
        }
    };
}