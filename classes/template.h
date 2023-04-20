#include <iostream>
#include "tensor.h"
using namespace std;
#pragma once

// generic class type for all layers ie convolutional, fcl, bath norm...
class Layer {
public:
    virtual ~Layer() { // virtual destructor
        cout << "Base destructor called." << endl;
    }
     virtual Tensor<double> forward(Tensor<double> input) = 0;
    virtual Tensor<double> backwards(Tensor<double> input) = 0;
};