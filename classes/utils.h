
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include "tensor.h"
#include "template.h"

using namespace std;

#pragma once

void printArray(Tensor<double> arr, int size) {
    for(int i=0; i<size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

void print_vector(const vector<vector<vector<double>>>& vec) {
   for (const auto& outer : vec) {
        for (const auto& middle : outer) {
            for (const auto& inner : middle) {
                std::cout << inner << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
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

void print_tensor(Tensor<double> &t) {
    cout << "[";
    for (int i = 0; i < t.depth; i++) {
        cout << "\n  [";
        for (int j = 0; j < t.rows; j++) {
            cout << "\n    [";
            for (int k = 0; k < t.cols; k++) {
                cout << t(i, j, k);
                if (k < t.cols - 1) {
                    cout << ", ";
                }
            }
            cout << "]";
            if (j < t.depth - 1) {
                cout << ",";
            }
        }
        cout << "\n  ]";
        if (i < t.filter_num - 1) {
            cout << ",";
        }
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


pair<int, double> get_pred(Tensor<double> outputs) {
    double max = -10000000;
    int idx = -1;
    int k = 0;
    while (k < outputs.size) {
        if (outputs[k] > max) {
            max = outputs[k];
            idx = k;
        }
        k++;
    }
    // cout << idx << endl;
    // assert(idx >= 0);
    return make_pair(idx, max);
}

class Flatten : public Layer { // assumes we always flatten down on forward pass and reshape on backpass
public:
    // string tag = "fltn";
    int input_depth;
    int intput_rows;
    int input_cols;

    Flatten() : Layer("fltn" ) {}

    Tensor<double> forward(Tensor<double> input_3d) {
        input_depth = input_3d.depth;
        intput_rows = input_3d.rows;
        input_cols = input_3d.cols;
        // cout << input_depth << endl;

        Tensor<double> flat(input_depth * intput_rows * input_cols);

        int idx = 0;
        for (int f = 0; f < input_depth; f++) {
            for (int r = 0; r < intput_rows; r++) {
                for (int c = 0; c < input_cols; c++) {
                    flat[idx] = input_3d(f, r, c);
                    idx++;
                }
            }
        }

        return flat;
    }
    
    Tensor<double> backwards(Tensor<double> input) {
        Tensor<double> output(input_depth, intput_rows, input_cols);

        int counter = 0;
        for(int d=0; d < input_depth; d++){
            for(int r=0; r < intput_rows; r++){
                for(int c=0; c < input_cols; c++){
                    output(d, r, c) = input[counter];
                    counter++;
                }
            }
        }
        return output;
    }

    int prune(){
        return 0;
    }

    
};