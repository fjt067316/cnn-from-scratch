#include <iostream>
#include "tensor.h"
using namespace std;
#pragma once

// generic class type for all layers ie convolutional, fcl, bath norm...
class Layer {
public:
    const string tag;

    Layer(string t) : tag(t) {}

    virtual ~Layer() { // virtual destructor
        cout << "Base destructor called." << endl;
    }
    virtual Tensor<double> forward(Tensor<double> input) = 0;
    virtual Tensor<double> backwards(Tensor<double> input) = 0;
    virtual int prune() = 0;

    const string get_tag (){
        return this->tag;
    }
};