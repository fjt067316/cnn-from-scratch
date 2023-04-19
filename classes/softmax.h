#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include "template.h"
using namespace std;

#pragma once

class Softmax : public Layer {
public:
vector<double> last_input;

    // Softmax(){

    // }

    vector<double> forward(vector<double> input) {
        last_input = input;
        double max = -100000;
        for (int i = 0; i < input.size(); i++) {
            max = (max > input[i]) ? max : input[i];
        }

        double sum = 0.0;
        for (int i = 0; i < input.size(); i++) {
            sum += exp(input[i]-max);
        }

        for (int i = 0; i < input.size(); i++) {
            input[i] = exp(input[i]-max) / sum;
        }
        
        return input;
    }

    vector<double> backwards(vector<double> dLdZ){ // cross_entropy dLdZ = -1/p
    // https://github.com/AlessandroSaviolo/CNN-from-Scratch/blob/master/src/layer.py
    // https://victorzhou.com/blog/intro-to-cnns-part-2/
        vector<double> dLdZ_exp(dLdZ.size(), 0.0);
        vector<double> dout_dt(dLdZ.size(), 0.0); // dout_dt is dLdZ next layer
        double sum_exp = 0.0;
        int label_idx;

        for(int i=0; i < last_input.size(); i++){
            dLdZ_exp[i] = exp(last_input[i]); 
            sum_exp += dLdZ_exp[i];       
            if(dLdZ[i] < 0){ // it will be negative
                label_idx = i;
            }
        }
        // i is the label index
        for(int i=0; i < last_input.size(); i++){
            dout_dt[i] = -dLdZ_exp[label_idx]*dLdZ_exp[i] / (sum_exp*sum_exp);
        }
        
        dout_dt[label_idx] = dLdZ_exp[label_idx] * (sum_exp - dLdZ_exp[label_idx]) / (sum_exp * sum_exp);

        for(int i=0; i < last_input.size(); i++){
            dout_dt[i] *= dLdZ[label_idx];
        }
        return dout_dt;
    }

};