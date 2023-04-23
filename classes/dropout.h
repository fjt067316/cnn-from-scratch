#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include "template.h"
using namespace std;

#pragma once


class Dropout : public Layer { // input and output are same dimensions
public:
    double p;
    binomial_distribution<double> dis;
    random_device rd;
    mt19937 gen;
    bool on_backprop;

    Dropout(float probability_zero, bool on_backprop) : 
        p(probability_zero),
        gen(rd()),
        dis(0, p) // https://cplusplus.com/reference/random/binomial_distribution/binomial_distribution/
    {}
    
    Tensor<double> forward(Tensor<double> input) { // highest input dimension is 3
        
        int depth = (input.depth > 0) ? input.depth : 1;
        int rows = (input.rows > 0) ? input.rows : 0;
        int cols = (input.cols > 0) ? input.cols : input.size;

        for(int d=0; d < depth; d++){
            for(int k=0; k < rows; k++){
                for(int l=0; l < cols; l++){
                    input(d, k, l) *= dis(gen);
                }
            }
        }
        return input;
    }

    Tensor<double> backwards(Tensor<double> dLdZ){
        if(!on_backprop){
            return dLdZ;
        }
        int depth = (dLdZ.depth > 0) ? dLdZ.depth : 1;
        int rows = (dLdZ.rows > 0) ? dLdZ.rows : 0;
        int cols = (dLdZ.cols > 0) ? dLdZ.cols : dLdZ.size;

        for(int d=0; d < depth; d++){
            for(int k=0; k < rows; k++){
                for(int l=0; l < cols; l++){
                    dLdZ(d, k, l) *= dis(gen);
                }
            }
        }
        return dLdZ;
    }

    int prune(){
        return 0;
    }
};