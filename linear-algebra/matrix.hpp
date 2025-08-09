#pragma once

#include <vector>
#include <stdexcept>
#include <string>

namespace LinearAlgebra {
    class Matrix {
        private:
        std::vector<Vector> *matrix;

        public:
        // Constructors
        Matrix(const size_t rows, const size_t columns, const double defaultValue = 0) {
            matrix = new std::vector<Vector>(
                columns,
                Vector(rows, defaultValue)
            );
        }

        Matrix(const Matrix &other) {
            matrix = new std::vector<Vector>(
                other.noCols(),
                Vector(other.noRows())
            );

            for(size_t column = 0; column < other.noCols(); ++column) {
                for(size_t row = 0; row < other.noRows(); ++row) {
                    matrix->at(column).at(row) = other.at(row, column);
                }
            }
        }


        // Square Brackets
        Vector & operator[](size_t i) {return this->matrix->at(i);}
        Vector & at(size_t i) {return this->matrix->at(i);}
        Vector operator[](size_t i) const {return this->matrix->at(i);}
        Vector at(size_t i) const {return this->matrix->at(i);}

        double & at(size_t row, size_t column) {return this->at(column).at(row);}
        double at(size_t row, size_t column) const {return this->at(column).at(row);}


        // Size
        size_t noRows() const {
            return this->at(0).size();
        }
        size_t noCols() const {
            return this->matrix->size();
        }


        // Matrix Addition
        Matrix operator+(const Matrix &other) const {
            if(this->noRows() != other.noRows() || this->noCols() != other.noCols()) {
                throw std::invalid_argument("Both matricies must be the same size");
            }

            Matrix ret(*this);

            for(size_t column = 0; column < ret.noCols(); ++column) {
                for(size_t row = 0; row < ret.noRows(); ++row) {
                    ret.at(row, column) += other.at(row, column);
                }
            }

            return ret;
        }

        // Scalar Multiplication
        Matrix operator*(const double scalar) const {
            Matrix ret(*this);

            for(size_t columns = 0; columns < ret.noCols(); ++columns) {
                for(size_t row = 0; row < ret.noRows(); ++row) {
                    ret.at(row, columns) *= scalar;
                }
            }

            return ret;
        }
        friend Matrix operator*(const double scalar, const Matrix &matrix) {
            return matrix * scalar;
        }

        // Matrix Multiplication
        Matrix operator*(const Matrix &other) const {
            if(this->noCols() != other.noRows()) {
                throw std::invalid_argument("Can't multiply matricies, sizes don't match properly");
            }

            Matrix ret(this->noRows(), other.noCols());

            for(size_t column = 0; column < ret.noCols(); ++column) {
                for(size_t row = 0; row < ret.noRows(); ++row) {
                    size_t m = this->noCols(); // = other.noRows()
                    double sum = 0;

                    for(size_t k = 0; k < m; ++k) {
                        sum += this->at(row, k) * other.at(k, column);
                    }

                    ret.at(row, column) = sum;
                }
            }

            return ret;
        }


        // Identity
        static Matrix id(const size_t n) {
            Matrix ret(n, n);

            for(size_t i = 0; i < n; ++i) {
                ret.at(i, i) = 1;
            }

            return ret;
        }

        // String
        std::string string() const {
            std::string ret = "(";
            
            for(size_t row = 0; row < this->noRows(); ++row) {
                ret += "(";

                for(size_t column = 0; column < this->noCols(); ++column) {
                    ret += std::to_string(this->at(row, column)) + " ";
                }

                ret += ")\n";
            }

            ret.pop_back();
            ret += ")";

            return ret;
        }

        ~Matrix() {
            delete this->matrix;
        }
    };
}