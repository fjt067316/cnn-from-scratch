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
// #include "model.h"
using namespace std;

int working();
int heavy();
int custom();
int lr();
// The function that will be executed in each thread
void threadFunction(int threadId) {
    std::cout << "Thread " << threadId << " is running" << std::endl;
}


vector<double> decodeCsvString(string csv){
    std::vector<double> values;
    std::stringstream ss(csv);
    std::string item;

    while (getline(ss, item, ',')) {
        values.push_back(std::stoi(item));
    }

    // for (int value : values) {
    //     std::cout << value << " ";
    // }
    return values;
}

template 
class Tensor<int>;

template
class Tensor<float>;

template
class Tensor<double>;

int main(){
    // working();

    // Out of range error
    //t1d(0, 0);
    //t2d(0, 0, 0);

    return 0;
    // Model* model = new Model();
    // model->add_conv_layer(8, 1, 3);
    // model->add_max_pool(2,2);
    
    return 1;
}

// https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist
int heavy() { 
    // 784 - [32C5-P2] - [64C5-P2] - 128

    ConvolutionLayer* conv_32C5 = new ConvolutionLayer(32, 1, 5, 1); // 32x24x24
    Pool* max_pool_32 = new Pool(2,2); // 32x12x12
    ConvolutionLayer* conv_64C5 = new ConvolutionLayer(64, 32, 5, 1); // 64x8x8
    BatchNorm3D* bn = new BatchNorm3D(64);

    Pool* max_pool_64 = new Pool(2,2); // 64x4x4

    FullyConnectedLayer* fc_1024_128 = new FullyConnectedLayer(64*4*4, 128); // 1024
    FullyConnectedLayer* fc_128_10 = new FullyConnectedLayer(128, 10); // 1024

    Softmax* softmax = new Softmax();

    ifstream inputFile("./data/mnist_train.csv");

    // Check if the file was opened successfully
    if (!inputFile.is_open()) {
        cerr << "Error: could not open file" << endl;
        return 1;
    }
    string row;
    getline(inputFile, row); // discard first header row
    double learning_rate = 0.001; //LinearLRScheduler(0.2, -0.000005)
    int n = 10000; // number of rows to read
    int num_correct = 0;
    int loss = 0;

    for (int i = 0; i < n; i++) {
        row.erase();
        getline(inputFile, row);
        vector<double> input = decodeCsvString(row); // input = (784)

        // if(i%1000 == 0){ // learning scheduler
        //     if(learning_rate < 0.00001){
        //         continue;
        //     }
        //     learning_rate *= 0.95;
        //     // learning_rate += -0.00005;
        // }

        int image_label = input.front(); // correct image value
        input.erase(input.begin());

        vector<vector<vector<double>>> image_1x28x28 = reshape_input(input, 28, 28); // image_1x28x28: 1x28x28

        vector<vector<vector<double>>> conv_32C5_out = conv_32C5->forward(image_1x28x28); // conv_32C5_out: 32x24x24
        // bn->batch_normalize(&conv_8C3_out);
        vector<vector<vector<double>>> max_pool_32_out = max_pool_32->max_pool(conv_32C5_out); // max_pool_32_out: 32x12x12
        dropout3d(0.1, max_pool_32_out);

        vector<vector<vector<double>>> conv_64C5_out = conv_64C5->forward(max_pool_32_out); // conv_64C5_out: 64x8x8 
        bn->forwards(&conv_64C5_out);
        vector<vector<vector<double>>> max_pool_64_out = max_pool_64->max_pool(conv_64C5_out); // conv_8C3_2_out: 64x4x4

        vector<double> flattened_pool = flatten(max_pool_64_out); // flattened_conv_64C3: 1024

        vector<double> fc_1024_128_out = fc_1024_128->forward(flattened_pool, true); // fcl_1024_10: 128
        vector<double> fc_128_10_out = fc_128_10->forward(flattened_pool); // fcl_1024_10: 128

        // printArray(fc_128_10_out, 10);
        vector<double> outputs = softmax->forward(fc_128_10_out);
        // printArray(outputs, 10);

        vector<double> dLdZ(outputs.size(), 0.0);
        double max = -10000000;
        int idx = -1;
        int k = 0;
        while(k < 10){
            if(outputs[k] > max){
                max = outputs[k];
                idx = k;
            }
            k++;
        }

        if ((i + 1) % 100 == 0) {
            printf("[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%\n", i + 1, static_cast<float>(loss) / 100, num_correct);
            loss = 0;
            num_correct = 0;
        }

        num_correct += (idx == image_label) ? 1 : 0;
        loss += -log(outputs[image_label]);
        // cout << i << ", " << bool(image_label==idx) << ", " << outputs[image_label] << endl;
        // copy(outputs.begin(), outputs.end(), dLdZ.begin()); // copy coutputs to dLdZ
        // dLdZ[image_label]--; // substract one hot encoded label for initial dL/dZ
        dLdZ[image_label] = -1 / outputs[image_label];
        // printArray(dLdZ, 10);
        vector<double> dLdZ_softmax = softmax->backwards(dLdZ);
        // printArray(dLdZ_softmax, 10);
        vector<double> dLdZ_128 = fc_128_10->backwards(dLdZ_softmax);
        vector<double> dLdZ_1024 = fc_1024_128->backwards(dLdZ_128);
        // printArray(dLdZ_1352, 10);
        // print_vector(fc_1024_128->weights);
        vector<vector<vector<double>>> dLdZ_6444 = reshape(dLdZ_1024, 64, 4, 4); // dLdZ_138: 64x4x4
        vector<vector<vector<double>>> dLdZ_6488 = max_pool_64->upsample(dLdZ_6444); // dLdZ_1612: 64x8x8
        // print_vector(dLdZ_6488);
        bn->backwards(&conv_64C5_out);
        vector<vector<vector<double>>> dLdZ_3212 = conv_64C5->backwards(dLdZ_6488); //dLdZ_3212: 32x12x12
        vector<vector<vector<double>>> dLdZ_3224 = max_pool_32->upsample(dLdZ_3212); // dLdZ_3224:  32x24x24
        vector<vector<vector<double>>> dLdZ_final = conv_32C5->backwards(dLdZ_3224); //dLdZ_final: 1x28x28

    }
    return 0;

}


int working() {

    ConvolutionLayer* conv_8C3 = new ConvolutionLayer(8, 1, 3, 1); // 8x26x26
    BatchNorm3D* bn = new BatchNorm3D(8);
    Pool* max_pool = new Pool(2,2);
    FullyConnectedLayer* fc_1352_10 = new FullyConnectedLayer(8*13*13, 10);
    Softmax* softmax = new Softmax();

    ifstream inputFile("./data/mnist_train.csv");

    // Check if the file was opened successfully
    if (!inputFile.is_open()) {
        cerr << "Error: could not open file" << endl;
        return 1;
    }
    string row;
    getline(inputFile, row); // discard first header row
    double learning_rate = 0.01; //LinearLRScheduler(0.2, -0.000005)
    int n = 20000; // number of rows to read
    int num_correct = 0;
    int loss = 0;
    for (int i = 0; i < n; i++) {
        row.erase();
        getline(inputFile, row);
        vector<double> input = decodeCsvString(row); // input = (784)
        if(i%1000 == 0){
            if(learning_rate < 0.00001){
                continue;
            }
            // learning_rate *= 0.95;
            learning_rate += -0.0005;
        }

        int image_label = input.front(); // correct image value
        input.erase(input.begin());

        vector<vector<vector<double>>> image_1x28x28 = reshape_input(input, 28, 28); // image_1x28x28: 1x28x28

        vector<vector<vector<double>>> conv_8C3_out = conv_8C3->forward(image_1x28x28); // conv_8C3_1_out: 8x26x26
        // dropout3d(0.4, conv_8C3_out);
        bn->forwards(&conv_8C3_out);

        vector<vector<vector<double>>> max_pool_out = max_pool->max_pool(conv_8C3_out); // conv_8C3_2_out: 8x24x24

        vector<double> flattened_pool = flatten(max_pool_out); // flattened_conv_64C3: 1024

        vector<double> fc_1352_10_out = fc_1352_10->forward(flattened_pool, true); // fcl_1024_128: 128x
        // printArray(fc_1352_10_out, 10);
        vector<double> outputs = softmax->forward(fc_1352_10_out);
        // printArray(outputs, 10);

        vector<double> dLdZ(outputs.size(), 0.0);
        double max = -10000000;
        int idx = -1;
        int k = 0;
        while(k < 10){
            if(outputs[k] > max){
                max = outputs[k];
                idx = k;
            }
            k++;
        }

        if ((i + 1) % 100 == 0) {
            printf("[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%\n", 
                   i + 1, static_cast<float>(loss) / 100, num_correct);
            loss = 0;
            num_correct = 0;
        }

        num_correct += (idx == image_label) ? 1 : 0;
        loss += -log(outputs[image_label]);
        // cout << i << ", " << bool(image_label==idx) << ", " << outputs[image_label] << endl;
        // copy(outputs.begin(), outputs.end(), dLdZ.begin()); // copy coutputs to dLdZ
        // dLdZ[image_label]--; // substract one hot encoded label for initial dL/dZ
        dLdZ[image_label] = -1 / outputs[image_label];
        // printArray(dLdZ, 10);
        vector<double> dLdZ_softmax = softmax->backwards(dLdZ);
        // printArray(dLdZ_softmax, 10);
        vector<double> dLdZ_1352 = fc_1352_10->backwards(dLdZ_softmax);
        // printArray(dLdZ_1352, 10);
        // print_vector(fc_1352_10->weights);
        vector<vector<vector<double>>> dLdZ_138 = reshape(dLdZ_1352, 8, 13, 13); // dLdZ_138: 8x13x13

        vector<vector<vector<double>>> dLdZ_826 = max_pool->upsample(dLdZ_138); // dLdZ_1612: 8x26x26
        bn->backwards(&dLdZ_826);
        // printArray(dLdZ_826[0][0], 10);
        vector<vector<vector<double>>> dLdZ_final = conv_8C3->backwards(dLdZ_826); //dLdZ_final: 1x28x28
        // print_vector(conv_8C3->filters[0]);
    }
    return 0;

}


int custom() {

    ConvolutionLayer* conv_8C3 = new ConvolutionLayer(8, 1, 3, 1); // 8x26x26
    BatchNorm3D* bn_8 = new BatchNorm3D(8);
    ConvolutionLayer* conv_16C5 = new ConvolutionLayer(16, 8, 5, 1); // 8x22x22
    BatchNorm3D* bn_16 = new BatchNorm3D(16);
    FullyConnectedLayer* fc_7744_256 = new FullyConnectedLayer(16*22*22, 256); 
    BatchNorm1D* bn_1d = new BatchNorm1D();
    FullyConnectedLayer* fc_256_10 = new FullyConnectedLayer(256, 10);
    Softmax* softmax = new Softmax();

    ifstream inputFile("./data/mnist_train.csv");

    // Check if the file was opened successfully
    if (!inputFile.is_open()) {
        cerr << "Error: could not open file" << endl;
        return 1;
    }
    string row;
    getline(inputFile, row); // discard first header row

    double k = 0.01;
    int n = 10000; // number of rows to read
    int num_correct = 0;
    int loss = 0;
    for (int i = 0; i < n; i++) {
        row.erase();
        getline(inputFile, row);
        vector<double> input = decodeCsvString(row); // input = (784)


        int image_label = input.front(); // correct image value
        input.erase(input.begin());

        vector<vector<vector<double>>> image_1x28x28 = reshape_input(input, 28, 28); // image_1x28x28: 1x28x28

        vector<vector<vector<double>>> conv_8C3_out = conv_8C3->forward(image_1x28x28); // conv_8C3_1_out: 8x26x26
        // dropout3d(0.4, conv_8C3_out);
        // bn_8->batch_normalize(&conv_8C3_out);

        vector<vector<vector<double>>> conv_16C3_out = conv_16C5->forward(conv_8C3_out); // conv_16C3_out: 16x22x22
        // bn_16->batch_normalize(&conv_16C3_out);

        vector<double> flattened_conv = flatten(conv_16C3_out); // flattened_conv_64C3: 1024

        vector<double> fc_7744_256_out = fc_7744_256->forward(flattened_conv, true); // fcl_1024_128: 128x
        // printArray(fc_1352_10_out, 10);
        // bn_1d->batch_normalize(&fc_7744_256_out);
        vector<double> fc_256_10_out = fc_256_10->forward(fc_7744_256_out); // fcl_1024_128: 128x

        vector<double> outputs = softmax->forward(fc_256_10_out);
        // printArray(outputs, 10);

        vector<double> dLdZ(outputs.size(), 0.0);
        double max = -10000000;
        int idx = -1;
        int k = 0;
        while(k < 10){
            if(outputs[k] > max){
                max = outputs[k];
                idx = k;
            }
            k++;
        }

        if ((i + 1) % 100 == 0) {
            printf("[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%\n", 
                   i + 1, static_cast<float>(loss) / 100, num_correct);
            loss = 0;
            num_correct = 0;
        }

        num_correct += (idx == image_label) ? 1 : 0;
        loss += -log(outputs[image_label]);
        // cout << i << ", " << bool(image_label==idx) << ", " << outputs[image_label] << endl;
        // copy(outputs.begin(), outputs.end(), dLdZ.begin()); // copy coutputs to dLdZ
        // dLdZ[image_label]--; // substract one hot encoded label for initial dL/dZ
        dLdZ[image_label] = -1 / outputs[image_label];
        // printArray(dLdZ, 10);
        vector<double> dLdZ_softmax = softmax->backwards(dLdZ);
        // printArray(dLdZ_softmax, 10);
        vector<double> dLdZ_256 = fc_256_10->backwards(dLdZ_softmax);
        // dLdZ_256 = bn_1d->backwards(dLdZ_256, learning_rate);
        vector<double> dLdZ_1352 = fc_7744_256->backwards(dLdZ_256);
        // printArray(dLdZ_1352, 10);
        // print_vector(fc_1352_10->weights);
        vector<vector<vector<double>>> dLdZ_822 = reshape(dLdZ_1352, 16, 22, 22); // dLdZ_138: 8x13x13
        // bn_16->backwards(&dLdZ_822, learning_rate);
        vector<vector<vector<double>>> dLdZ_16 = conv_16C5->backwards(dLdZ_822); //dLdZ_final: 1x28x28
        // bn_8->backwards(&dLdZ_16, learning_rate);
        vector<vector<vector<double>>> dLdZ_final = conv_8C3->backwards(dLdZ_16); //dLdZ_final: 1x28x28
    }
    return 0;

}

// int lr() {

//     ConvolutionLayer* conv_8C3 = new ConvolutionLayer(8, 1, 3, 1); // 8x26x26
//     BatchNorm3D* bn = new BatchNorm3D(8);
//     Pool* max_pool = new Pool(2,2);
//     FullyConnectedLayer* fc_1352_10 = new FullyConnectedLayer(8*13*13, 10);
//     Softmax* softmax = new Softmax();

//     ifstream inputFile("./data/mnist_train.csv");

//     // Check if the file was opened successfully
//     if (!inputFile.is_open()) {
//         cerr << "Error: could not open file" << endl;
//         return 1;
//     }
//     string row;
//     getline(inputFile, row); // discard first header row
//     double learning_rate = 0.01; //LinearLRScheduler(0.2, -0.000005)
//     double k=0.01;
//     int n = 50000; // number of rows to read
//     int num_correct = 0;
//      double new_learning_rate = learning_rate;
//     int loss = 0;
//     for (int i = 0; i < n; i++) {
//         row.erase();
//         getline(inputFile, row);
//         vector<double> input = decodeCsvString(row); // input = (784)
//         if(i%1000 == 0){
//             if(learning_rate < 0.0001){
//                 continue;
//             }
//             int t = i/100;
//             new_learning_rate = learning_rate * exp(-k*t);
//         }

//         int image_label = input.front(); // correct image value
//         input.erase(input.begin());

//         vector<vector<vector<double>>> image_1x28x28 = reshape_input(input, 28, 28); // image_1x28x28: 1x28x28

//         vector<vector<vector<double>>> conv_8C3_out = conv_8C3->forward(image_1x28x28); // conv_8C3_1_out: 8x26x26
//         // dropout3d(0.4, conv_8C3_out);
//         bn->batch_normalize(&conv_8C3_out);

//         vector<vector<vector<double>>> max_pool_out = max_pool->max_pool(conv_8C3_out); // conv_8C3_2_out: 8x24x24

//         vector<double> flattened_pool = flatten(max_pool_out); // flattened_conv_64C3: 1024

//         vector<double> fc_1352_10_out = fc_1352_10->forward(flattened_pool, true); // fcl_1024_128: 128x
//         // printArray(fc_1352_10_out, 10);
//         vector<double> outputs = softmax->forward(fc_1352_10_out);
//         // printArray(outputs, 10);

//         vector<double> dLdZ(outputs.size(), 0.0);
//         double max = -10000000;
//         int idx = -1;
//         int k = 0;
//         while(k < 10){
//             if(outputs[k] > max){
//                 max = outputs[k];
//                 idx = k;
//             }
//             k++;
//         }

//         if ((i + 1) % 100 == 0) {
//             printf("[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%\n", 
//                    i + 1, static_cast<float>(loss) / 100, num_correct);
//             loss = 0;
//             num_correct = 0;
//         }

//         num_correct += (idx == image_label) ? 1 : 0;
//         loss += -log(outputs[image_label]);
//         // cout << i << ", " << bool(image_label==idx) << ", " << outputs[image_label] << endl;
//         // copy(outputs.begin(), outputs.end(), dLdZ.begin()); // copy coutputs to dLdZ
//         // dLdZ[image_label]--; // substract one hot encoded label for initial dL/dZ
//         dLdZ[image_label] = -1 / outputs[image_label];
//         // printArray(dLdZ, 10);
//         vector<double> dLdZ_softmax = softmax->backwards(dLdZ);
//         // printArray(dLdZ_softmax, 10);
//         vector<double> dLdZ_1352 = fc_1352_10->backwards(dLdZ_softmax, new_learning_rate);
//         // printArray(dLdZ_1352, 10);
//         // print_vector(fc_1352_10->weights);
//         vector<vector<vector<double>>> dLdZ_138 = reshape(dLdZ_1352, 8, 13, 13); // dLdZ_138: 8x13x13

//         vector<vector<vector<double>>> dLdZ_826 = max_pool->upsample(dLdZ_138); // dLdZ_1612: 8x26x26
//         bn->backwards(&dLdZ_826, new_learning_rate);
//         // print_vector(dLdZ_826);
//         vector<vector<vector<double>>> dLdZ_final = conv_8C3->backwards(dLdZ_826, new_learning_rate); //dLdZ_final: 1x28x28
//         // print_vector(conv_8C3->filters[0]);
//     }
//     return 0;

// }