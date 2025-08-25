default:
	g++ main.cpp -o main.exe -std=c++20 -I linear-algebra/ -I machine-learning
	./main.exe

darwin:
	clang++ main.cpp -o main -std-c++20 -I linear-algabr/ -I machine-learning
	./main