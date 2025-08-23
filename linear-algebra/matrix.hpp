#pragma once

#include<vector>
#include<cmath>
#include<string>
#include<stdexcept>


namespace LinearAlgebra {
    class Matrix {
        private:
        std::vector<double> matrix; // List of matrix ROWS
        size_t noRows, noColumns;

        public:
        Matrix();

        Matrix(const size_t rows, const size_t columns, double defaultValue = 0.0) {
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
        
        double & operator[] (size_t i) {
            if(i > noRows*noColumns) throw std::invalid_argument("Out of range");

            return this->matrix[i];
        }
        double operator[] (size_t i) const {
            if(i >= noRows*noColumns) throw std::invalid_argument("Out of range");

            return this->matrix[i];
        }


        size_t rows()    const { return this->noRows;    }
        size_t columns() const { return this->noColumns; }


        // Matrix Addition
        Matrix operator+(const Matrix &other) const {
            if(this->noRows != other.rows() || this->noColumns != other.columns()) {
                throw std::invalid_argument("Both Matricies must be the same size");
            }

            Matrix ret(*this);

            for(size_t i = 0; i < noRows * noColumns; ++i) {
                ret[i] += other[i];
            }

            return ret;
        }

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
            if(this->noColumns != other.rows()) { throw std::invalid_argument("Can't multiply, sizes don't match"); }

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
    };
}