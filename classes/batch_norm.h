#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include "template.h"
using namespace std;

#pragma once

class BatchNorm1D : public Layer { // input and output are same dimensions
// https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/ 
public:
    double gamma;
    double beta;
    vector<double> inputs;
    double epsilon;
    double variance;

    BatchNorm1D( double epsilon = 1e-5){
        // Initialize gamma and beta as 1 and 0 to start
        this->gamma = 1;
        this->beta = 0;
        this->epsilon = epsilon;
        // random_device rd;
        // mt19937 gen(rd());
        // uniform_real_distribution<double> dis(-1.0, 1.0);

        // for (int i = 0; i < input_shape; i++) {
        //     gamma[i] = dis(gen);
        //     beta[i] = dis(gen);
        // }
    }

    void batch_normalize(vector<double> *input) { // epsilon helps prevent division by 0
        int size = (*input).size();
        this->inputs = (*input);
        // Compute mean
        double mean = 0;
        for (int i = 0; i < size; i++) {
            mean += (*input)[i];
        }
        mean /= size;

        // Compute variance
        variance = 0;
        for (int i = 0; i < size; i++) {
            double diff = (*input)[i] - mean;
            variance += diff * diff;
        }
        variance /= size;

        // Compute standard deviation
        double stddev = sqrt(variance);

        // Normalize input
        for (int i = 0; i < size; i++) {
            (*input)[i] = ((*input)[i] - mean) / (stddev + epsilon);
        }

        // rescale 
        for (int i = 0; i < size; i++) {
            (*input)[i] = gamma*(*input)[i] + beta;
        }

    }

    vector<double> backwards(vector<double> dLdZ, double learning_rate){
        // https://deepnotes.io/batchnorm#backward-propagation
        vector<double> dLdA(dLdZ.size(), 0.0); //= dLdZ * gamma / sqrt(variance+epsilon); // epsilon added to avoid division by zero
        double dLdG = 0;
        double dLdB = 0;

        for(int i=0; i<dLdZ.size(); i++){
            dLdA[i] = dLdZ[i] * gamma / sqrt(variance+epsilon);
        }

        for(int i=0; i<dLdZ.size(); i++){
            dLdG += dLdZ[i] * inputs[i];
        }
        // dLdB == sum(dLdZ)
        for(int i=0; i<dLdZ.size(); i++){
            dLdB += dLdZ[i];
        }

        beta -= learning_rate * dLdB;
        gamma -= learning_rate * dLdG;

        return dLdA;
    }
};

class BatchNorm3D : public Layer { // input and output are same dimensions
public:
    vector<double> gamma;
    vector<double> beta;
    vector<vector<vector<double>>> inputs;
    double epsilon;
    vector<double> variance;
    int num_channels;

    BatchNorm3D(int num_channels, double epsilon = 1e-5){
        // 1 gamma and 1 beta per channel
        // Initialize gamma and beta as 1 and 0 to start
        this->gamma = vector<double>(num_channels, 1);
        this->beta = vector<double>(num_channels, 0);
        this->variance = vector<double>(num_channels, 0);
        this->num_channels = num_channels;
        this->epsilon = epsilon;
    }

    void batch_normalize(vector<vector<vector<double>>> *input) { // epsilon helps prevent division by 0
        int channel_rows = (*input)[0].size();
        int channel_cols = (*input)[0][0].size();

        int total = channel_cols * channel_rows;

        for(int idx=0; idx < num_channels; idx++){
        
            this->inputs = (*input);
            // Compute mean
            double mean = 0;
            for (int i = 0; i < channel_rows; i++) {
                for(int j=0; j<channel_cols; j++){
                    mean += (*input)[idx][i][j];
                }
            }

            mean /= total;

            // Compute variance
            variance[idx] = 0;
            for (int i = 0; i < channel_rows; i++) {
                for(int j=0; j<channel_cols; j++){
                    double diff = (*input)[idx][i][j] - mean;
                    variance[idx] += diff * diff;
                }
            }

            variance[idx] /=  total;

            // Compute standard deviation
            double stddev = sqrt(variance[idx]);

            // Normalize input
            for (int i = 0; i < channel_rows; i++) {
                for(int j=0; j<channel_cols; j++){
                    (*input)[idx][i][j] = ((*input)[idx][i][j]-mean) / (stddev + epsilon);
                }
            }

            for (int i = 0; i < channel_rows; i++) {
                for(int j=0; j<channel_cols; j++){
                    (*input)[idx][i][j] = gamma[idx]*(*input)[idx][i][j] + beta[idx];
                }
            }

        }

    }

    void backwards(vector<vector<vector<double>>> *dLdZ, double learning_rate){
        // https://deepnotes.io/batchnorm#backward-propagation
        // vector<vector<vector<double>>> dLdA(dLdZ.size(), vector<vector<double>>(dLdZ[0].size(), vector<double>(dLdZ[0][0].size(), 0.0)));
        vector<double> dLdG((*dLdZ).size(), 0.0);
        vector<double> dLdB((*dLdZ).size(), 0.0);
       
        for(int idx=0; idx < num_channels; idx++){
            for(int i=0; i<(*dLdZ)[0].size(); i++){
                for(int j=0; j<(*dLdZ)[0][0].size(); j++){
                    dLdG[idx] += (*dLdZ)[idx][i][j] * inputs[idx][i][j];
                }
            }
            for(int i=0; i<(*dLdZ)[0].size(); i++){
                for(int j=0; j<(*dLdZ)[0][0].size(); j++){
                    dLdB[idx] += (*dLdZ)[idx][i][j];
                }
            }

            for(int i=0; i<(*dLdZ)[0].size(); i++){
                for(int j=0; j<(*dLdZ)[0][0].size(); j++){
                    (*dLdZ)[idx][i][j] = (*dLdZ)[idx][i][j] * gamma[idx] / sqrt(variance[idx]+epsilon);
                }
            }
        }

        for(int i=0; i<(*dLdZ).size(); i++) {
            beta[i] -= learning_rate * dLdB[i];
            gamma[i] -= learning_rate * dLdG[i];
        }
    }
};