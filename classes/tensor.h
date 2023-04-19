#include <vector>
#include <stdexcept>
#include <iostream>
using namespace std;

#pragma once

template <typename T>
class Tensor {
public:
    Tensor(int size1) : data(size1), size(data.size()) {}
    Tensor(int size1, int size2) : data(size1 * size2), rows(size1), cols(size2), size(data.size()) {}
    Tensor(int size1, int size2, int size3) : data(size1 * size2 * size3), depth(size1), rows(size2), cols(size3), size(data.size()) {}
    Tensor(int size1, int size2, int size3, int size4) : data(size1 * size2 * size3 * size4), filter_num(size1), depth(size2), rows(size3), cols(size4), size(data.size()) {}

    T& operator[](int i) {
        if (size == 0) {
            throw out_of_range("Tensor is not a 1D vector");
        }
        return data[i];
    }

    T& operator()(int i, int j) {
        if ((rows == 0) || (cols == 0) ) {
            throw out_of_range("Tensor is not a 2D vector");
        }
        return data[i * rows + j];
    }

    T& operator()(int i, int j, int k) {
        if (depth == 0) {
            throw out_of_range("Tensor is not a 3D vector");
        }
        return data[i * rows * cols + j * cols + k];
    }

    T& operator()(int i, int j, int k, int l) {
        if (filter_num == 0) {
            throw out_of_range("Tensor is not a 4D vector");
        }
        return data[i * (depth * rows * cols) + j * (rows * cols) + k * cols + l];
    }


public:
    vector<T> data;
    int size = 0;
    int rows = 0;
    int cols = 0;
    int depth = 0;
    int filter_num = 0;
};


/*

#include <iostream>

int main() {
    Tensor<int> t1d(10);
    t1d[0] = 42;
    std::cout << t1d[0] << std::endl;

    Tensor<int> t2d(3, 4);
    t2d(1, 2) = 99;
    std::cout << t2d(1, 2) << std::endl;

    Tensor<int> t3d(2, 3, 4);
    t3d(0, 1, 2) = 123;
    std::cout << t3d(0, 1, 2) << std::endl;

    // Out of range error
    //t1d(0, 0);
    //t2d(0, 0, 0);

    return 0;
}

*/