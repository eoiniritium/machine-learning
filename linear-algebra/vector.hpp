#pragma once

#include <vector>
#include <stdexcept>
#include <string>

namespace LinearAlgebra {
    class Vector {
        private:
        std::vector<double> *vector;

        public:
        // Constructors
        Vector(const size_t n, const double defaultValue = 0) {
            vector = new std::vector<double>(n, defaultValue);
        }

        Vector(const std::vector<double> &vector) {
            this->vector = new std::vector<double>(vector.size());

            for(size_t i = 0; i < vector.size(); ++i) {
                this->vector->at(i) = vector[i];
            }
        }

        Vector(const Vector &other) {
            size_t n = other.size();

            this->vector = new std::vector<double>(n);
            for(size_t i = 0; i < n; ++i) {
                this->vector->at(i) = other[i];
            }
        }


        // Square Brackets
        double & operator [](size_t i) {return this->vector->at(i);}
        double & at(size_t i) {return this->vector->at(i);}
        double operator[] (size_t i) const {return this->vector->at(i);}
        double at(size_t i) const {return this->vector->at(i);}


        // Vector Addition
        Vector operator+(const Vector &other) const {
            if(this->size() != other.size()) throw std::invalid_argument("Both vectors must be of the same dimension");
            
            Vector ret(*this);

            for(size_t i = 0; i < this->size(); ++i) {
                ret[i] += other[i];
            }

            return ret;
        }


        // Scalar Multiplication
        Vector operator*(const double scalar) const {
            Vector ret(*this);

            for(size_t i = 0; i < ret.size(); ++i) {
                ret[i] *= scalar;
            }

            return ret;
        }
        friend Vector operator*(const double scalar, const Vector &vector) {
            return vector * scalar;
        }


        // Dot Product
        double dot(const Vector &other) const {
            if(other.size() != this->size()) throw std::invalid_argument("Both vectors must be of the same dimension");
            
            double dotProduct = 0;

            for(size_t i = 0; i < this->size(); ++i) {
                dotProduct += this->at(i) * other[i]; 
            }

            return dotProduct;
        }


        size_t size() const {
            return this->vector->size();
        }

        std::string string() const {
            std::string ret = "(";
            
            for(size_t i = 0; i < this->size(); ++i) {
                ret += std::to_string(this->at(i)) + " ";
            }

            ret.pop_back();

            return ret + ")";
        }

        ~Vector() {
            delete this->vector;
        }
    };
}