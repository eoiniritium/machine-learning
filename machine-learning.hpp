#include "linear-algebra/vector.hpp"
#include "linear-algebra/matrix.hpp"

class NeuralNetwork {
    private:
    size_t inputSize, hiddenSize, outputSize;

    
    public:
    NeuralNetwork(size_t inputSize, size_t hiddenSize, size_t outputSize) {
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;

        
    }
};