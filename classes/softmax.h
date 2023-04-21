#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cassert>
#include "template.h"
#include "tensor.h"
using namespace std;

#pragma once

class Softmax : public Layer {
public:
Tensor<double> last_input;

    Softmax(){
        return;
    }

    Tensor<double> forward(Tensor<double> input) { // 1d input
        assert((input.rows == 0) && (input.cols==0) && (input.depth==0)); // input is 1d
        last_input = input;
        double max = INT32_MIN;
        // input.print();

        for (int i = 0; i < input.size; i++) {
            max = (max > input[i]) ? max : input[i];
        }

        double sum = 0.0;
        for (int i = 0; i < input.size; i++) {
            sum += exp(input[i]-max);
        }

        for (int i = 0; i < input.size; i++) {
            input[i] = exp(input[i]-max) / sum;
        }

        return input;
    }

    Tensor<double> backwards(Tensor<double> dLdZ){ // cross_entropy dLdZ = -1/p
    // https://github.com/AlessandroSaviolo/CNN-from-Scratch/blob/master/src/layer.py
    // https://victorzhou.com/blog/intro-to-cnns-part-2/
        assert((dLdZ.depth == 0) && (dLdZ.rows == 0) && (dLdZ.cols==0)); // assert dLdZ is 1d
        // dLdZ.print();
        Tensor<double> dLdZ_exp(dLdZ.size);
        Tensor<double> dout_dt(dLdZ.size); // dout_dt is dLdZ next layer
        double sum_exp = 0.0;
        int label_idx;

        for(int i=0; i < last_input.size; i++){
            dLdZ_exp[i] = exp(last_input[i]); 
            sum_exp += dLdZ_exp[i];       
            if(dLdZ[i] < 0){ // it will be negative
                label_idx = i;
            }
        }
        // i is the label index
        for(int i=0; i < last_input.size; i++){
            dout_dt[i] = -dLdZ_exp[label_idx]*dLdZ_exp[i] / (sum_exp*sum_exp);
        }
        
        dout_dt[label_idx] = dLdZ_exp[label_idx] * (sum_exp - dLdZ_exp[label_idx]) / (sum_exp * sum_exp);

        for(int i=0; i < last_input.size; i++){
            dout_dt[i] *= dLdZ[label_idx];
        }
        return dout_dt;
    }

};