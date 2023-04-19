#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
using namespace std;
#pragma once

void relu(Tensor<double> *input_3d, bool derivative=false) {
    if(derivative){
        for (int i = 0; i < input_3d->depth; i++){
            for (int j = 0; j < input_3d->rows; j++){
                for (int k = 0; k < input_3d->cols; k++){
                    (*input_3d)(i, j, k) = ((*input_3d)(i, j, k) > 0) ? 1 : 0;
                }
            }
        }
    } else {
        for (int i = 0; i < input_3d->depth; i++){
            for (int j = 0; j < input_3d->rows; j++){
                for (int k = 0; k < input_3d->cols; k++){
                    // (*input_3d)[i][j][k] = max((*input_3d)[i][j][k],0.001*(*input_3d)[i][j][k]); // leaky relu
                    (*input_3d)(i, j, k) = max((*input_3d)(i, j, k),0.0); // normal relu
                }
            }
        }
    }
}

void relu(vector<double> *input, bool derivative=false) {
    if(derivative){
        for (int i = 0; i < input->size(); i++){
            (*input)[i] = ((*input)[i] > 0) ? 1 : 0;
        }
    } else {
        for (int i = 0; i < input->size(); i++){
            (*input)[i] = max((*input)[i], 0.001*(*input)[i]);
        }
    }
}