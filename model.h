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
using namespace std;

// generic type for layers so we dont have to use void* 


class Model{
    public:
    vector<Layer*> model;

    Model() {
        return;
    }

    void train(vector<vector<vector<double>>> image_input, int iterations, int learning_rate){
        
        for (int i = 0; i < model.size(); ++i) {
            Layer layer = *model[i];

         }
    }

    void add_conv_layer(int num_filters, int input_depth, int filter_len, int stride = 1, bool padding=0){
        ConvolutionLayer * conv = new ConvolutionLayer(num_filters, input_depth, filter_len, stride, padding);
        model.push_back(conv);
        return;
    }

    void add_fcl_layer(int input_size, int output_size){
        FullyConnectedLayer* fcl = new FullyConnectedLayer(input_size, output_size);
        model.push_back(fcl);
    }

    void add_max_pool(int pool_size, int stride){
        Pool* max_pool = new Pool(pool_size, stride);
        model.push_back(max_pool);
    }

    void add_batch_norm_1D(){
        BatchNorm1D* bn = new BatchNorm1D();
        model.push_back(bn);
    }

    void add_batch_norm_3D(int filter_depth){
        BatchNorm3D* bn = new BatchNorm3D(filter_depth);
        model.push_back(bn);
    }

    void add_softmax(){
        Softmax* softmax = new Softmax();
        model.push_back(softmax);
    }
};
