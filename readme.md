# Machine Learning
My Machine Learning project --- all from scratch. Written using only the C++20 Standard Library.

## Training
1) Initialise an instance of MachineLearning::NeuralNetwork either by loading a model from file or from scratch
2) Load training data from file.
3) Specify hyper-parameters (Learning Rate, Epochs, Batch Size, Output Frequency)
4) Train Model.

Example of training an XOR gate.

training.txt:

	0,0 0
	0,1 1
	1,0 1
	1,1 0

main.cpp:

	const double learningRate = 0.3;
	const size_t epochs = 1e4;
	const size_t batchSize = 1;
	const size_t outputFrequency = 1000;
	
	// leakyRELU, leakyRELUPrime, costPrime implementation HERE
	
	int main() {
		MachineLearning::NeuralNetwork net(
			{2, 3, 1},
			leakyRELU,
			leakyRELUPrime,
			costPrime
		);
	
		auto trainingData = MachineLearning::loadTrainingData("train.txt");
		net.train(
			trainingData,
			batchSize,
			epochs,
			learningRate,
			outputFrequency
		);
		
		LinearAlgebra::Matrix input(2, 1);
		input.at(0, 0) = 1;
		input.at(1, 0) = 0;
		
		std::cout << net.predict(input).string() << std::endl;
		
		return 0;
	}

## Saving a Model
Once a model is trained, it is possible to save the model to file.

	MachineLearning::writeModel(net, "model.txt");

## Loading a Model
You can then initialise a model with weights and biases from file.

	MachineLearning::NeuralNetwork net(
		MachineLearning::loadModel("model.txt"),
		leakyRELU,
		leakyRELUPrime,
		costPrime
	);