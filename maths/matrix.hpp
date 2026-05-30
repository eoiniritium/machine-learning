#pragma once

#include <vector>
#include <utility>
#include <string>
#include <stdexcept>
#include <format>

namespace Maths {
    template<class T>
    class Matrix {
        protected:
        size_t rows, cols;
        std::vector<T> mtx;

        public:
        Matrix() {}

        Matrix(size_t rows, size_t cols) {
            this->rows = rows;
            this->cols = cols;

            mtx = std::vector<T>(rows * cols, 0);
        }

        Matrix(std::vector<std::vector<T>> matrix) {
            rows = matrix.size();
            cols = matrix[0].size();

            mtx = std::vector<T>(rows * cols);

            size_t i = 0;
            for (auto row : matrix) {
                for(auto element: row) {
                    mtx[i++] = element;
                }
            }
        }

        Matrix(const Matrix &matrix) {
            auto shape = matrix.shape();
            rows = shape.first;
            cols = shape.second;

            mtx = std::vector<T>(rows * cols);

            for(size_t i = 0; i < rows; ++i) {
                for(size_t j = 0; j < cols; ++j) {
                    mtx[idx(i, j)] = matrix[i, j];
                }
            }
        }

        static Matrix I(size_t n) {
            auto identity = Matrix<T>(n, n);

            for(size_t i = 0; i < n; ++i) {
                identity[i, i] = static_cast<T>(1.0f);
            }

            return identity;
        }

        T operator[] (size_t i, size_t j) const { return mtx[idx(i, j)]; }
        T &operator[] (size_t i, size_t j) { return mtx[idx(i, j)]; }

        // Addition
        Matrix operator+(const Matrix &other) const {
            if(shape() != other.shape()) {
                throw std::invalid_argument(std::format(
                    "Matricies must be same shape\n"
                    "Attempting ({}x{})+({}x{})",
                    rows, cols, other.shape().first, other.shape().second
                ));
            }

            auto ret = Matrix<T>(rows, cols);

            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; ++j) {
                    ret[i, j] = mtx[idx(i, j)] + other[i, j];
                }
            }

            return ret;
        }

        // Multiplication
        Matrix operator*(const T &scalar) const {
            auto ret = Matrix<T>(rows, cols);

            for(size_t i = 0; i < rows; ++i) {
                for(size_t j = 0; j < cols; ++j) {
                    ret[i, j] = scalar * mtx[idx(i, j)];
                }
            }

            return ret;
        }
        friend Matrix operator*(const T &scalar, const Matrix &matrix) {
            return matrix * scalar;
        }

        Matrix operator*(const Matrix &other) const {
            auto otherShape = other.shape();

            if(cols != otherShape.first) {
                throw std::invalid_argument(std::format(
                    "Matricies must be same shape\n"
                    "Attempting ({}x{})*({}x{})",
                    rows, cols, other.shape().first, other.shape().second
                ));
            }

            auto ret = Matrix<T>(rows, otherShape.second);


            for(size_t i = 0; i < rows; ++i) {
                for(size_t j = 0; j < otherShape.second; ++j) {
                    T sum = static_cast<T>(0);

                    for(size_t k = 0; k < cols; ++k) {
                        sum += mtx[idx(i, k)] * other[k, j];
                    }

                    ret[i, j] = sum;
                }
            }

            return ret;
        }

        std::pair<size_t, size_t> shape() const { return std::pair<size_t, size_t>(rows, cols); }

        friend std::ostream& operator << ( std::ostream& outs, const Matrix<T> &matrix) {
            std::string matrixString = "|";

            size_t i = 0;
            while (i < matrix.rows * matrix.cols - 1 ) {
                matrixString += std::to_string(matrix.mtx[i++]) + " ";
                if(i % matrix.cols == 0) {
                    matrixString += "|\n|";
                }
            }

            return outs << matrixString << std::to_string(matrix.mtx.back()) << " |";
        }

        Matrix transpose() {
            auto ret = Matrix<T>(cols, rows);

            for(size_t i = 0; i < rows; ++i) {
                for(size_t j = 0; j < cols; ++j) {
                    ret[j, i] = mtx[idx(i, j)];
                }
            }

            return ret;
        }

        Matrix repeatColumn(size_t nCols) {
            if(cols != 1) {
                throw std::invalid_argument("Matrix must be a column vector.");
            }

            auto columnValues = transpose().asVector()[0];

            auto vectorRepeated = std::vector<std::vector<double>>(nCols);
            for(size_t i = 0; i < nCols; ++i) {
                vectorRepeated[i] = columnValues;
            }

            return Matrix(vectorRepeated).transpose();
        }

        std::vector<std::vector<T>> asVector() {
            return mtx;
        }

        protected:
        size_t idx(size_t i, size_t j) const { return i*cols + j; }
    };
}