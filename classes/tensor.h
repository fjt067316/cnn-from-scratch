#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>
using namespace std;

#pragma once


template <typename T>
void printArray(vector<T> arr) {
    for(int i=0; i<arr.size(); i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

template <typename T>
class Tensor {
public:
    Tensor() : data(0), size(0), rows(0), cols(0), depth(0), dims(0), filter_num(0) {}
    Tensor(int size1) : data(size1, 0), dims(1), size(data.size()) {}
    Tensor(int size1, int size2) : data(size1 * size2, 0), rows(size1), cols(size2), dims(2), size(data.size()) {}
    Tensor(int size1, int size2, int size3) : data(size1 * size2 * size3, 0), depth(size1), rows(size2), cols(size3), dims(3), size(data.size()) {}
    Tensor(int size1, int size2, int size3, int size4) : data(size1 * size2 * size3 * size4, 0), filter_num(size1), depth(size2), rows(size3), cols(size4), dims(4), size(data.size()) {}

    T& operator[](int i) {
        if (size == 0) {
            throw out_of_range("Tensor is not a 1D vector");
        }
        return data[i];
    }

    T& operator()(int i) {
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
        if (dims != 3) {
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

    void push_back(const T& val) {
        data.push_back(val);
        size = data.size();
        return;
    }

    // add item to tensor at given indices
    // void operator = (const <t> &Value ) { 
    //      feet = D.feet;
    //      inches = D.inches;
    //   }
    void operator()(int i, int j, const T& val) {
        if ((rows == 0) || (cols == 0) ) {
            throw out_of_range("Tensor is not a 2D vector");
        }
        data[i * rows + j] = val;
    }
    
    void operator()(int i, int j, int k, const T& val) {
        if (depth == 0) {
            throw out_of_range("Tensor is not a 3D vector");
        }
        // cout << val << endl;
        data[i * rows * cols + j * cols + k] = val;
    }
    
    void operator()(int i, int j, int k, int l, const T& val) {
        if (filter_num == 0) {
            throw out_of_range("Tensor is not a 4D vector");
        }
        data[i * (depth * rows * cols) + j * (rows * cols) + k * cols + l] = val;
    }

    // void operator()(int i, int j, int k, const T& val) { // DO NOT DELETE THIS SHIT WORKS LIKE MAGIC FOR ASSIGNMENTS   t(1, 0, 3) = 7;
    //     if (depth == 0) {
    //         throw out_of_range("Tensor is not a 3D vector");
    //     }
    //     cout << "hit" << endl;
    //     data[i * rows * cols + j * cols + k] = val;
    // }
    
    // void operator()(int i, int j, int k, int l, const T& val) {
    //     if (filter_num == 0) {
    //         throw out_of_range("Tensor is not a 4D vector");
    //     }
    //     data[i * (depth * rows * cols) + j * (rows * cols) + k * cols + l] = val;
    // }


    void zero() {
        for(int i=0; i < data.size(); i++){
            data[i] = 0;
        }
    }
    

    void print() {
        printArray(data);
        if (dims == 1) {
            for (int i = 0; i < size; i++) {
                cout << data[i] << " ";
            }
            cout << endl;
        } else if (dims == 2) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    cout << data[i * rows + j] << " ";
                }
                cout << endl;
            }
        } else if (dims == 3) {
            for (int i = 0; i < depth; i++) {
                for (int j = 0; j < rows; j++) {
                    for (int k = 0; k < cols; k++) {
                        cout << data[i * rows * cols + j * cols + k] << " ";
                    }
                    cout << endl;
                }
                cout << endl;
            }
        } else {
            throw out_of_range("Tensor has more than 3 dimensions");
        }
    }


public:
    vector<T> data;
    int size = 0;
    int rows = 0;
    int cols = 0;
    int depth = 0;
    int filter_num = 0;
    int dims = 0;
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

