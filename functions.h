#include <iostream>

template<typename T>
void print(const T& message) {
    std::cout << message << std::endl;  // Automatically appends a newline, mimicking Python's print
}

