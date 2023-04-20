#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
using namespace std;

#pragma once

void dropout3d(double probability, vector<vector<vector<double>>>& inputVector)
{
    // Initialize random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);

    // Iterate over each element in the 3D vector and zero out with given probability
    for (auto& i : inputVector) {
        for (auto& j : i) {
            for (auto& k : j) {
                double randomNum = dis(gen);
                if (randomNum < probability) {
                    k = 0;
                }
            }
        }
    }
}


void he_weight_init(vector<vector<vector<vector<double>>>> *filters, int size)
{// https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    default_random_engine generator;
    normal_distribution<double> distribution(0,  sqrt(2.0 / size));

    // set the entries of the vector to a random value
    for (int i = 0; i < filters->size(); i++)
    {
        for (int j = 0; j < (*filters)[0].size(); j++)
        {
            for (int k = 0; k < (*filters)[0][0].size(); k++)
            {
                for (int l = 0; l < (*filters)[0][0][0].size(); l++)
                {
                    (*filters)[i][j][k][l] = distribution(generator);

                    // cout << distribution(generator) << endl;
                }
            }
        }
    }
}


void he_weight_init(vector<vector<vector<double>>> *bias, int size)
{// https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    default_random_engine generator;
    normal_distribution<double> distribution(0,  sqrt(2.0 / size));

    // set the entries of the vector to a random value
    for(int i = 0; i < (*bias).size(); i++) {
        for(int j = 0; j < (*bias)[0].size(); j++) {
            for(int k=0; k < (*bias)[0][0].size(); k++){
                (*bias)[i][j][k] = distribution(generator);
            }
        }
    }
}

void he_weight_init(vector<double> *bias, int size)
{// https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    default_random_engine generator;
    normal_distribution<double> distribution(0,  sqrt(2.0 / size));

    // set the entries of the vector to a random value
    for(int i = 0; i < (*bias).size(); i++) {
        (*bias)[i] = distribution(generator);
    }
}

void set_mask(vector<double> *mask) {
    random_device rd;
    mt19937 gen(rd());
    binomial_distribution<> d(1, 0.6);

    for(int i = 0; i < mask->size(); i++) {
        (*mask)[i] = d(gen);
    }
}