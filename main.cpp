#include <iostream>
#include <thread>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include "classes/conv.h"
#include "classes/softmax.h"
#include "classes/fcl.h"
#include "classes/pool.h"
#include "classes/utils.h"
#include "classes/zeros.h"
#include "classes/batch_norm.h"
#include "classes/activation_functions.h"
#include "classes/tensor.h"
#include "model.h"
using namespace std;

int working();
int heavy();
int custom();
int model();


vector<double> decodeCsvString(string csv){
    vector<double> values;
    stringstream ss(csv);
    string item;

    while (getline(ss, item, ',')) {
        values.push_back(std::stoi(item));
    }

    // for (int value : values) {
    //     std::cout << value << " ";
    // }
    return values;
}

// template 
// class Tensor<int>;

// template
// class Tensor<float>;

// template
// class Tensor<double>;

int main(){
    // working();
    model();

    return 1;
}


int model(){

    Model* model = new Model();
    model->add_conv_layer(8, 1, 3, 0.001); // 1x28x28 -> 8x24x24
    // model->add_conv_layer(8, 8, 5, 0.001); // 1x28x28 -> 8x20x20
    model->add_batch_norm_3D(8, 0.001);
    model->add_max_pool(2,2); // 8x24x24 -> 8x12x12
    // model->add_conv_layer(16, 8, 5, 0.0001); // 8x12x12 -> 16x8x8
    // model->add_batch_norm_3D(16, 0.001);
    // model->add_max_pool(2,2); // 16x8x8 -> 16x4x4
    model->add_flatten(); // 
    model->add_fcl_layer(8*13*13, 10, 0.0005);
    // model->add_batch_norm_1D(0.001);
    // model->add_fcl_layer(256, 10, 0.01);
    model->add_softmax();

    ifstream inputFile("./data/mnist_train.csv");
    if (!inputFile.is_open()) {
        cerr << "Error: could not open file" << endl;
        return 1;
    }
    string row;
    getline(inputFile, row); // discard first header row

    int iterations = 30000;

    int num_correct = 0;
    float loss = 0;
    for(int n=0; n<iterations; n++){
        row.erase();
        getline(inputFile, row);
        vector<double> input = decodeCsvString(row); // input = (784)

        int image_label = input.front(); // correct image value
        input.erase(input.begin());
    // cout << "hit" << endl;
        Tensor<double> image_1x28x28 = reshape_input(input, 28, 28); // 8x28 -> 1x28x28
        num_correct += model->forward(image_1x28x28, image_label, &loss);

        if ((n + 1) % 100 == 0) {
            printf("[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%\n", n + 1, static_cast<float>(loss) / 100, num_correct);
            loss = 0;
            num_correct = 0;
        }
    }

    return 1;
}



int working(){

    Model* model = new Model();
    model->add_conv_layer(8, 1, 3, 0.0001, 1); // 1x28x28 -> 8x26x26
    model->add_batch_norm_3D(8, 0.0001);
    model->add_conv_layer(12, 8, 5, 0.0005, 1); // 8x26x26 -> 12x22x22
    model->add_batch_norm_3D(12, 0.0001);
    model->add_max_pool(2,2); //  12x22x22 ->  12x11x11
    // model->add_conv_layer(16, 8, 5, 0.0001); // 8x12x12 -> 16x8x8
    // model->add_batch_norm_3D(16, 0.001);
    // model->add_max_pool(2,2); // 16x8x8 -> 16x4x4
    model->add_flatten(); // 
    model->add_fcl_layer(12*11*11, 256, 0.0005, 1);
    model->add_batch_norm_1D(0.0005);
    model->add_fcl_layer(256, 10, 0.0005, 1);
    model->add_softmax();

    ifstream inputFile("./data/mnist_train.csv");

    if (!inputFile.is_open()) {
        cerr << "Error: could not open file" << endl;
        return 1;
    }
    string row;
    getline(inputFile, row); // discard first header row

    int iterations = 10000;

    int num_correct = 0;
    float loss = 0;
    for(int n=0; n<iterations; n++){
        row.erase();
        getline(inputFile, row);
        vector<double> input = decodeCsvString(row); // input = (784)

        int image_label = input.front(); // correct image value
        input.erase(input.begin());
    // cout << "hit" << endl;
        Tensor<double> image_1x28x28 = reshape_input(input, 28, 28); // 8x28 -> 1x28x28
        num_correct += model->forward(image_1x28x28, image_label, &loss);

        if ((n + 1) % 100 == 0) {
            printf("[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%\n", n + 1, static_cast<float>(loss) / 100, num_correct);
            loss = 0;
            num_correct = 0;
        }
    }

    return 1;
}

// https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist
