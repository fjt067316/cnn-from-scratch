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
#include "transformer.h"
#include "model.h"
using namespace std;

int working();
int kaggle();
int transformer();
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
    // kaggle();
    // model();
    transformer();

    // Testing save and load 

    // Model* model = new Model();
    // model->add_conv_layer(8, 1, 5, 0.001); // 1x28x28 -> 8x24x24
    // model->add_fcl_layer(28*28*8, 256);
    // model->save();
    // model->load();

    return 1;
}

/*
784 - [32C3-32C3-32C5S2] - [64C3-64C3-64C5S2] - 128 - 10
with 40% dropout, batch normalization, and data augmentation added
*/
// https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist
int kaggle(){

    Model* model = new Model();
    model->add_conv_layer(8, 1, 5, 0.001); // 1x28x28 -> 8x24x24
    model->add_batch_norm_3D(8, 0.0001);
    model->add_max_pool(2,2); // 8x24x24-> 8x12x12
    model->add_conv_layer(16, 8, 5, 0.001); // 8x12x12-> 16x8x8
    model->add_batch_norm_3D(8, 0.0001);
    model->add_dropout(0.4); // 1x28x28 -> 8x20x20

    // model->add_batch_norm_3D(16, 0.001);
    model->add_flatten(); // 
    model->add_fcl_layer(16*8*8, 128, 0.001);
    model->add_batch_norm_1D(0.0001);
    model->add_dropout(0.4); // 1x28x28 -> 8x20x20
    model->add_fcl_layer(128, 10, 0.001);
    model->add_softmax();

    ifstream inputFile("./data/mnist_train.csv");
    if (!inputFile.is_open()) {
        cerr << "Error: could not open file" << endl;
        return 1;
    }
    string row;
    getline(inputFile, row); // discard first header row

    int iterations = 30;

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

// prune every 5000 iterations then continue to train with pruned model
int model(){

    int factor = 0;
    int num_prunes = 4;

    Model* model = new Model();
    model->add_conv_layer(8, 1, 3, 0.0001); // 1x28x28 -> 8x26x26
    model->add_batch_norm_3D(8, 0.0001);
    model->add_max_pool(2,2); // 8x26x26 -> 8x13x13
    // model->add_conv_layer(16, 8, 5, 0.0001); // 8x12x12 -> 16x8x8
    // model->add_batch_norm_3D(16, 0.001);
    // model->add_max_pool(2,2); // 16x8x8 -> 16x4x4
    model->add_flatten(); // 
    model->add_fcl_layer(8*13*13, 128, 0.0005);
    model->add_batch_norm_1D(0.0001);
    model->add_fcl_layer(128, 10, 0.0001);
     model->add_batch_norm_1D(0.0001);

    model->add_softmax();

    ifstream inputFile("./data/mnist_train.csv");
    if (!inputFile.is_open()) {
        cerr << "Error: could not open file" << endl;
        return 1;
    }
    string row;
    getline(inputFile, row); // discard first header row

    int iterations = 20000;

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

        if( (num_prunes > 0) && (n+1) % (3000) == 0){
            cout << "pruned: " << model->prune() << endl;
            factor = 1500;
            num_prunes--;
        }
    }

    return 1;
}



int working(){

    Model* model = new Model();
    model->add_conv_layer(8, 1, 3, 0.001, 1); // 1x28x28 -> 8x24x24
    model->add_batch_norm_3D(8, 0.001);
    // model->add_conv_layer(12, 8, 5, 0.001); // 8x24x24 -> 12x20x20
    // model->add_batch_norm_3D(12, 0.001);
    model->add_max_pool(2,2); //  12x24x24 ->  12x12x12
    // model->add_conv_layer(16, 8, 5, 0.0001); // 8x12x12 -> 16x8x8
    // model->add_batch_norm_3D(16, 0.001);
    // model->add_max_pool(2,2); // 16x8x8 -> 16x4x4
    model->add_flatten(); // 
    model->add_fcl_layer(8*13*13, 10, 0.0005, 1);
    // model->add_batch_norm_1D(0.001);
    // model->add_fcl_layer(256, 10, 0.001, 1);
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

void experiment(){
    auto start = std::chrono::high_resolution_clock::now();
    int a = 0;
    int b = 12;
    for(int i=0; i<200000; i++){
        // a -= 0.01*b;
        // a = fma(0.01, b, a);
        b--;
        a++;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
}

// https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist

// #include <unordered_map>

int transformer() {
    // Load the glove6b.txt file into a hash table
    unordered_map<string, vector<double>> hash_table = load_embedding_map("data/glove.6B/glove.6B.50d.txt"); 

    // Test the hash table
    string word = "hello";
    if (hash_table.find(word) != hash_table.end()) {
        vector<double> vec = hash_table[word];
        cout << "Vector for word " << word << ": [ ";
        for (double value : vec) {
            cout << value << " ";
        }
        cout << "]" << endl;
    } else {
        cout << "Word " << word << " not found in hash table." << endl;
    }
    string word2 = "penis";
    if (hash_table.find(word2) != hash_table.end()) {
        vector<double> vec = hash_table[word2];
        cout << "Vector for word " << word2 << ": [ ";
        for (double value : vec) {
            cout << value << " ";
        }
        cout << "]" << endl;
    } else {
        cout << "Word " << word2 << " not found in hash table." << endl;
    }

    return 0;
}