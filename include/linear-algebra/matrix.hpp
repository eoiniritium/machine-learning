#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <functional>
#include <format>
#include <random>
#include <utility>


namespace LinearAlgebra {
    class Matrix {
        private:
        std::vector<double> matrix; // List of matrix ROWS
        size_t noRows, noColumns;

        public:
        Matrix() {};

        Matrix(const size_t rows, const size_t columns, const std::pair<double, double> randomRange) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution distribution(randomRange.first, randomRange.second);

            this->noRows = rows;
            this->noColumns = columns;

            this->matrix = std::vector<double>(rows * columns, 0);

            for(size_t i = 0; i < rows * columns; ++i) {
                this->matrix[i] = distribution(gen);
            }
        };

        Matrix(const size_t rows, const size_t columns, const double defaultValue = 0.0) {
            this->noRows = rows;
            this->noColumns = columns;


            this->matrix = std::vector<double>(rows * columns, defaultValue);
        }

        Matrix(const std::vector<std::vector<double>> mat) {
            this->noRows = mat.size();
            this->noColumns = mat[0].size();

            this->matrix = std::vector<double>(noRows * noColumns);

            for(size_t row = 0; row < noRows; ++row) {
                for(size_t column = 0; column < noColumns; ++column) {
                    this->matrix[row * noColumns + column] = mat[row][column];
                }
            }
        }

        Matrix(const Matrix &other) {
            this->noRows = other.rows();
            this->noColumns = other.columns();

            this->matrix = std::vector<double>(this->noRows * this->noColumns);

            for(size_t i = 0; i < noRows * noColumns; ++i) {
                this->matrix[i] = other[i];
            }
        }

        double & at(size_t i, size_t j) {return this->matrix[i * noColumns + j];}
        double at(size_t i, size_t j) const {return this->matrix[i * noColumns + j];}

        double & operator[] (size_t i) { return this->matrix[i]; }
        double operator[] (size_t i) const { return this->matrix[i]; }


        size_t rows()    const { return this->noRows;    }
        size_t columns() const { return this->noColumns; }


        // Matrix Addition
        Matrix operator+(const Matrix &other) const {
            if(this->noRows != other.rows() || this->noColumns != other.columns()) {
                throw std::invalid_argument(
                    std::format(
                        "Addition/Subtraction: Both Matricies must be the same size. LHS: ({}x{}). RHS: ({}x{})",
                        this->noRows,
                        this->noColumns,
                        other.rows(),
                        other.columns()
                    )
                );
            }

            Matrix ret(*this);

            for(size_t i = 0; i < noRows * noColumns; ++i) {
                ret[i] += other[i];
            }

            return ret;
        }
        Matrix operator-(const Matrix &other) const { return (*this) + (-1.0) * other; }

        // Scalar Multiplication
        Matrix operator*(const double scalar) const {
            Matrix ret(*this);

            for(size_t i = 0; i < noRows * noColumns; ++i) {
                ret[i] *= scalar;
            }

            return ret;
        }
        friend Matrix operator*(const double scalar, const Matrix &other) { return other * scalar; }

        // Matrix Multiplication
        Matrix operator*(const Matrix &other) const {
            if(this->noColumns != other.rows()) {
                throw std::invalid_argument(
                    std::format(
                        "Matrix multiplication: Cant multiply ({}x{})x({}x{})",
                        this->noRows,
                        this->noColumns,
                        other.rows(),
                        other.columns()
                    )
                );
            }

            size_t newRows = this->noRows;
            size_t newCols = other.columns();
            size_t sumLimit = this->noColumns;
            Matrix ret(newRows, newCols);
            
            for(size_t row = 0; row < newRows; ++row) {
                for(size_t col = 0; col < newCols; ++col) {
                    double sum = 0;
                    for(size_t k = 0; k < sumLimit; ++k) {
                        sum += this->at(row, k) * other.at(k, col);
                    }

                    ret.at(row, col) = sum;
                }
            }

            return ret;
        }

        static Matrix id(const size_t n) {
            Matrix ret(n, n);

            for(size_t i = 0; i < n; ++i) {
                ret.at(i, i) = 1.0;
            }

            return ret;
        }

        Matrix transpose() const {
            Matrix ret(this->noColumns, this->noRows);

            for(size_t row = 0; row < this->noRows; ++row) {
                for(size_t col = 0; col < this->noColumns; ++col) {
                    ret.at(col, row) = this->at(row, col);
                }
            }

            return ret;
        }

        Matrix vectorise(std::function<double(double)> f) const {
            Matrix ret(*this);

            for(size_t i = 0; i < this->noRows * this->noColumns; ++i) {
                ret[i] = f(ret[i]);
            }

            return ret;
        }

        std::string string() const {
            std::string ret = "[";

            for(size_t row = 0; row < noRows; ++row) {
                ret += "[";

                for(size_t col = 0; col < noColumns; ++col) {
                    ret += std::to_string(this->at(row, col)) + " ";
                }

                ret += "]\n";
            }

            ret.pop_back();
            ret += "]";

            return ret;
        }

        Matrix extendFromRow(const size_t noRows) const {
            if(this->noRows != 1) {
                throw std::invalid_argument("Matrix is not of size 1xN");
            }

            Matrix ret(noRows, this->noColumns);

            for(size_t row = 0; row < noRows; ++row) {
                for(size_t col = 0; col < this->noColumns; ++ col) {
                    ret.at(row, col) = this->at(1, col);
                }
            }

            return ret;
        }
        Matrix extendFromColumn(const size_t noCols) const {
            if(this->noColumns != 1) {
                throw std::invalid_argument("Matrix is not of size 1xN");
            }

            Matrix ret(this->noRows, noCols);

            for(size_t row = 0; row < ret.rows(); ++row) {
                for(size_t col = 0; col < ret.columns(); ++col) {
                    ret.at(row, col) = this->at(row, 1);
                }
            }

            return ret;
        }

        Matrix hadamardProduct(const Matrix &other) const {
            if(this->rows() != other.rows() || this->columns() != other.columns()) {
                throw std::invalid_argument("Hadamard Product: Matricies are of different dimensions!");
            }

            Matrix ret(*this);

            for(size_t i = 0; i < ret.rows() * ret.columns(); ++i) {
                ret[i] *= other[i];
            }

            return ret;
        }

        double sumOverColumn(const size_t column) const {
            double sum = 0;

            for(size_t row = 0; row < noRows; ++row) {
                sum += this->at(row, column);
            }

            return sum;
        }
    };
}