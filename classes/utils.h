
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include "tensor.h"
using namespace std;

#pragma once

void printArray(vector<double> arr, int size) {
    for(int i=0; i<size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

void print_vector(const vector<vector<vector<double>>>& vec) {
    cout << "[";
    for (const auto& v1 : vec) {
        cout << "\n  [";
        for (const auto& v2 : v1) {
            cout << "\n    [";
            for (const auto& elem : v2) {
                cout << elem << ", ";
            }
            cout << "],";
        }
        cout << "\n  ],";
    }
    cout << "\n]\n";
}

void print_vector(const vector<vector<double>>& vec) {
    cout << "[";
    for (const auto& v : vec) {
        cout << "\n  [";
        for (const auto& elem : v) {
            cout << elem << ", ";
        }
        cout << "],";
    }
    cout << "\n]\n";
}

Tensor<double> reshape_input(vector<double> input_1d, int rows, int cols){
    
    Tensor<double> output(1, rows, cols); 
    int counter = 0;

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            output(0,i,j) = input_1d[counter];
            counter++;
        }
    }

    return output;
}


Tensor<double> reshape(vector<double> input, int depth, int rows, int cols) {
    Tensor<double> output(depth, rows,cols);
    int counter = 0;
    for(int d=0; d < depth; d++){
        for(int r=0; r < rows; r++){
            for(int c=0; c < cols; c++){
                output(d, r, c) = input[counter];
                counter++;
            }
        }
    }
    return output;
}

vector<double> flatten(vector<vector<vector<double>>> feature_map) {
    int num_filters = feature_map.size();
    int rows = feature_map[0].size();
    int cols = feature_map[0][0].size();

    vector<double> flat(num_filters * rows * cols);

    int idx = 0;
    for (int f = 0; f < num_filters; f++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                flat[idx++] = feature_map[f][r][c];
            }
        }
    }

    return flat;
}

vector<double> min_max_scale(vector<double> input) {
    // Find the minimum and maximum values in the input vector
    double min = *min_element(input.begin(), input.end());
    double max = *max_element(input.begin(), input.end());

    // Scale the input values to the range [0,1]
    for (int i = 0; i < input.size(); i++) {
        input[i] = (input[i] - min) / (max - min);
    }

    return input;
}

// 4d vector input
void print_shape(const vector<vector<vector<vector<double>>>> &vec)
{
    int a = vec.size();
    int b = vec[0].size();
    int c = vec[0][0].size();
    int d = vec[0][0][0].size();
    cout << "(" << a << ", " << b << ", " << c << ", " << d << ")" << endl;
}

// 3d vector input
void print_shape(const vector<vector<vector<double>>> &vec)
{
    int a = vec.size();
    int b = vec[0].size();
    int c = vec[0][0].size();
    cout << "(" << a << ", " << b << ", " << c << ")" << endl;
}
// 1d vector input
void print_shape(const vector<double> &vec)
{
    int a = vec.size();
    cout << "(" << a << ")" << endl;
}
