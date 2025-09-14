default:
	g++ main.cpp -o main.exe -std=c++20 -I include/
	./main.exe

darwin:
	clang++ main.cpp -o main.out -std=c++20 -I include/
	./main.out