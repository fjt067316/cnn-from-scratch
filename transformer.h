#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace std;

#pragma once

// Function to load the glove6b.txt file into a hash table
unordered_map<string, vector<double>> load_embedding_map(string filename) {
    unordered_map<string, vector<double>> hash_table;

    ifstream infile(filename);
    if (!infile) {
        cerr << "Error opening file " << filename << endl;
        return hash_table;
    }

    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        string word;
        ss >> word;

        vector<double> vec;
        double value;
        while (ss >> value) {
            vec.push_back(value);
        }

        hash_table[word] = vec;
    }

    infile.close();

    return hash_table;
}




// Attention 
// https://www.youtube.com/watch?v=W2rWgXJBZhU
/*
// https://glassboxmedicine.com/2019/08/15/the-transformer-attention-is-all-you-need/

nn.Embedding consists of a weight matrix W that will transform a one-hot vector into a real-valued vector.
*/

// https://jalammar.github.io/illustrated-transformer/ 
void get_word_encoding(string word){

}

/*

one hot encode words, as many words as inputs to fully connected layer
fully connected kayer has 512 outputs

*/