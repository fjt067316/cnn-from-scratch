#include <iostream>
#include <thread>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include "classes.h"
using namespace std;

// The function that will be executed in each thread
void threadFunction(int threadId) {
    std::cout << "Thread " << threadId << " is running" << std::endl;
}

vector<double> flatten(vector<vector<vector<double>>> feature_map) {
    int num_filters = feature_map.size();
    int rows = feature_map[0].size();
    int cols = feature_map[0][0].size();

    vector<double> flat(num_filters * rows * cols);

    int idx = 0;
    for (int f = 0; f < num_filters; f++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                flat[idx++] = feature_map[f][r][c];
            }
        }
    }

    return flat;
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

vector<vector<vector<double>>> reshape_input(vector<double> input_1d, int rows, int cols){
    
    vector<vector<vector<double>>> output(1, vector<vector<double>>(rows, vector<double>(cols, 0))); // fill matrix with 0's
    int counter = 0;

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            output[0][i][j] = input_1d[counter];
            counter++;
        }
    }

    return output;
}

int main() {

    ConvolutionLayer* conv_8C3 = new ConvolutionLayer(8, 28, 28, 1, 3, 1); // 8x26x26
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
    int n = 50000; // number of rows to read
    int num_correct = 0;
    int loss = 0;
    for (int i = 0; i < n; i++) {
        row.erase();
        getline(inputFile, row);
        vector<double> input = decodeCsvString(row); // input = (784)
        if(i%1000 == 0){
            if(learning_rate < 0.0001){
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
        bn->batch_normalize(&conv_8C3_out);

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
        vector<double> dLdZ_1352 = fc_1352_10->backwards(dLdZ_softmax, learning_rate);
        // printArray(dLdZ_1352, 10);
        // print_vector(fc_1352_10->weights);
        vector<vector<vector<double>>> dLdZ_138 = reshape(dLdZ_1352, 8, 13, 13); // dLdZ_138: 8x13x13

        vector<vector<vector<double>>> dLdZ_826 = max_pool->upsample(dLdZ_138); // dLdZ_1612: 8x26x26
        bn->backwards(&dLdZ_826, learning_rate);
        // print_vector(dLdZ_826);
        vector<vector<vector<double>>> dLdZ_final = conv_8C3->backwards(dLdZ_826, learning_rate); //dLdZ_final: 1x28x28
        // print_vector(conv_8C3->filters[0]);
    }
    return 0;

}
