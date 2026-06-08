default:
	g++ main.cpp -o main.exe -std=c++23
	./main.exe

darwin:
	clang++ main.cpp -o main.out -std=c++23
	./main.out