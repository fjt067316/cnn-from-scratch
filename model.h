#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include "classes/conv.h"
#include "classes/softmax.h"
#include "classes/fcl.h"
#include "classes/pool.h"
#include "classes/utils.h"
#include "classes/zeros.h"
#include "classes/batch_norm.h"
#include "classes/activation_functions.h"
#include "classes/tensor.h"
using namespace std;

#pragma once

// generic type for layers so we dont have to use void* 
template 
class Tensor<int>;

template
class Tensor<float>;

template
class Tensor<double>;

class Model{
    public:
    vector<Layer*> model;
    int size = 0;

    Model() {
        return;
    }

    int forward(Tensor<double> x, int image_label){ // int iterations, int learning_rate
        Layer* layer;
        
        
        Tensor<double> dLdZ(10);
        // for(int n=0; n < iterations; n++){
            for (int i = 0; i < size; ++i) {
                layer = model[i];
                x = layer->forward(x);
            }
            // x should now be size 10
            // printArray(x, 10);
            pair<int, double> pred = get_pred(x);
            int pred_idx = pred.first;
            double pred_val = pred.second;
            dLdZ.zero();
            dLdZ[image_label] = -1 / x[image_label];
            x = dLdZ;
            // cout << x[image_label] << endl;
            for (int i = size-1; i >= 0; --i) {
                layer = model[i];
                x = layer->backwards(x);
            }
        // }

        return pred_idx == image_label;

    }

    void add_conv_layer(int num_filters, int input_depth, int filter_len, double learning_rate=0.001, int stride = 1, bool padding=0){
        ConvolutionLayer * conv = new ConvolutionLayer(num_filters, input_depth, filter_len, learning_rate, stride, padding);
        model.push_back(conv);
        size++;
    }

    void add_fcl_layer(int input_size, int output_size, double learning_rate=0.001, bool dropout=false){
        FullyConnectedLayer* fcl = new FullyConnectedLayer(input_size, output_size, learning_rate, dropout);
        model.push_back(fcl);
        size++;
    }

    void add_max_pool(int pool_size, int stride){
        Pool* max_pool = new Pool(pool_size, stride);
        model.push_back(max_pool);
        size++;
    }

    void add_batch_norm_1D(){
        BatchNorm1D* bn = new BatchNorm1D();
        model.push_back(bn);
        size++;
    }

    void add_batch_norm_3D(int filter_depth, double learning_rate=0.001){
        BatchNorm3D* bn = new BatchNorm3D(filter_depth, learning_rate);
        model.push_back(bn);
        size++;
    }

    void add_flatten(){
        Flatten* flatten = new Flatten();
        model.push_back(flatten);
        size++;
    }

    void add_softmax(){
        Softmax* softmax = new Softmax();
        model.push_back(softmax);
        size++;
    }
};
