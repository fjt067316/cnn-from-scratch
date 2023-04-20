#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cassert>
#include "template.h"
#include "tensor.h"
using namespace std;

#include "zeros.h"
#include "utils.h"
#include "activation_functions.h"

#pragma once


class AdamConv{
    // https://datascience.stackexchange.com/questions/25024/strange-behavior-with-adam-optimizer-when-training-for-too-long
public:
    vector<vector<vector<vector<double>>>>  m_dw, v_dw; // one for every weight in layer
    vector<double> m_db, v_db; // one for every bias in layer
    double beta1, beta2, epsilon, learning_rate, initial_b1, initial_b2;
    int t = 0;
    int counter=0;
    int iterations = 100;
    double weight_decay = 0;
    int filter_depth, filter_size, num_filters;
    
    AdamConv(int num_filters, int filter_depth, int filter_size, double learning_rate, double beta1=0.9, double beta2=0.999, double epsilon=1e-8) :
    m_dw(num_filters, vector<vector<vector<double>>>(filter_depth, vector<vector<double>>(filter_size, vector<double>(filter_size, 0)))),
    v_dw(num_filters, vector<vector<vector<double>>>(filter_depth, vector<vector<double>>(filter_size, vector<double>(filter_size, 0)))),
    m_db(num_filters, 0),
    v_db(num_filters, 0)
    {
       
        this->beta1 = beta1;
        // this->initial_b1 = beta1;
        // this->initial_b2 = beta2;
        this->beta2 = beta2;
        this->epsilon = epsilon;
        this->learning_rate = learning_rate;
        this->filter_depth = filter_depth;
        this->num_filters = num_filters;
        this->filter_size = filter_size;
    }

    void update( vector<vector<vector<vector<double>>>> *w, vector<double> *b, vector<vector<vector<vector<double>>>> dw, vector<double> db) { // t is current timestep
        // dw, db are what we would usually update params with gradient descent
        // counter++;
        // if((counter % iterations) == 0){
        //     // beta1 *= initial_b1;
        //     // beta2 *= initial_b2;
        // }
        this->t++;


        // cout << dw[0][0][2][2] << endl;

        // momentum beta 1
        // weights
        for(int n=0; n < num_filters; n++){
            for(int d=0; d < filter_depth; d++){
                for(int j=0; j < filter_size; j++){
                    for(int k=0; k < filter_size; k++){
                        m_dw[n][d][j][k] = beta1 * m_dw[n][d][j][k] + (1 - beta1) * (dw[n][d][j][k] + weight_decay*(*w)[n][d][j][k]); // biased momentum estimate
                        v_dw[n][d][j][k] = beta2 * v_dw[n][d][j][k] + (1 - beta2) * pow((dw[n][d][j][k] + weight_decay*(*w)[n][d][j][k]), 2); // bias corrected momentum estimate     
                        // cout << m_dw[n][d][j][k] << endl;
                    }
                }
            }
        }
        // printArray(m_dw[0], 10);

        // cout << m_dw[7][7] << endl;
        // print_vector(m_dw);
        // biases
        for (int i = 0; i < (*w).size(); ++i) {
            m_db[i] = beta1 * m_db[i] + (1 - beta1) * (db[i] + weight_decay*(*b)[i]);
            v_db[i] = beta2 * v_db[i] + (1 - beta2) * pow((db[i] + weight_decay*(*b)[i]), 2);
        }

        

        // rms beta 2
        // weights
        // biases
        vector<vector<vector<vector<double>>>> m_dw_corr(num_filters, vector<vector<vector<double>>>(filter_depth, vector<vector<double>>(filter_size, vector<double>(filter_size, 0))));
        vector<vector<vector<vector<double>>>> v_dw_corr(num_filters, vector<vector<vector<double>>>(filter_depth, vector<vector<double>>(filter_size, vector<double>(filter_size, 0))));
        vector<double> m_db_corr(m_dw.size());
        vector<double> v_db_corr(m_dw.size());

        double denom_mw = (1 - pow(beta1, t));
        double denom_vw = (1 - pow(beta2, t));
        // cout << denom_mw << endl;
        for(int n=0; n < num_filters; n++){
            for(int d=0; d < filter_depth; d++){
                for(int j=0; j < filter_size; j++){
                    for(int k=0; k < filter_size; k++){
                        m_dw_corr[n][d][j][k] = m_dw[n][d][j][k] / denom_mw; // biased momentum estimate
                        v_dw_corr[n][d][j][k] = v_dw[n][d][j][k] / denom_vw; // bias corrected momentum estimate     
                    }
                }
            }
        }

        // bias correction
        double denom_mb = (1 - pow(beta1, t));
        double denom_vb = (1 - pow(beta2, t));

        for (int j = 0; j < m_dw.size(); j++) {
            m_db_corr[j] = m_db[j] / denom_mb;
            v_db_corr[j] = v_db[j] / denom_vb;
        }

        double clip_threshold = 0;

        // printArray(m_dw_corr[0], 10);
        // update weights and biases 
        // double gamma = gamma_init * decay_rate;
        for(int n=0; n < num_filters; n++){
            for(int d=0; d < filter_depth; d++){
                for(int j=0; j < filter_size; j++){
                    for(int k=0; k < filter_size; k++){
                        (*w)[n][d][j][k] -= learning_rate * (m_dw_corr[n][d][j][k] / (sqrt(v_dw_corr[n][d][j][k]) + epsilon)); 
                        if(clip_threshold > 0){
                            (*w)[n][d][j][k] = min(max((*w)[n][d][j][k], -clip_threshold), clip_threshold);
                        }
                    }
                }
            }
        }
        
        // printArray(m_db_corr, 10);
        // learning_rate = learning_rate * sqrt(denom_vb) / denom_mb;

        // b -= learning_rate *  dL/dZ
        for(int i=0; i < (*b).size(); i++){
            (*b)[i] -= learning_rate * (m_db_corr[i] / (sqrt(v_db_corr[i]) + epsilon)); 
            if(clip_threshold > 0){
                (*b)[i] = min(max((*b)[i], -clip_threshold), clip_threshold); 
            }
        }
        
        return;
    }
};


// class ConvolutionLayerDepthwise
// // https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9
// // https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/ 
// {
// public:
//     int filter_len;
//     int stride;
//     int num_filters;
//     bool padding;
//     vector<vector<vector<double>>> filters;
//     vector<vector<vector<double>>> _input;
//     vector<vector<vector<double>>> _output;
//     int intput_depth;


//     ConvolutionLayerDepthwise(int num_filters, int filter_len, int stride = 1, bool padding=0) {
//         this->num_filters = num_filters; // num_filters == input_depth
//         this->filter_len = filter_len;
//         this->stride = stride;
//         this->padding = padding;
        
//         // filters is shape (feature_map_num x input_3d_depth x filter_height x filter_width)
//         this->filters.resize(num_filters, vector<vector<vector<double>>>(input_depth, vector<vector<double>>(filter_len, vector<double>(filter_len, 0.0))));
//         // vector<vector<vector<vector<double>>>> filters(num_filters, vector<vector<vector<double>>>(input_depth, vector<vector<double>>(filter_len, vector<double>(filter_len, []() { return (double)rand() / RAND_MAX; }))));
//         fill_with_random(&(this->filters));
//     }

//     vector<vector<vector<double>>> forward(vector<vector<vector<double>>> input3d) {
//         //https://stackoverflow.com/questions/59887786/algorithim-of-how-conv2d-is-implemented-in-pytorch 
//         // Get the input volume dimensions

//         _input = input3d;
//         input_depth = input3d.size();

//         int input_rows = input3d[0].size();
//         int input_cols = input3d[0][0].size();

//         if(padding){
//             int padding_num = (filter_len - 1);
//             input_rows = input3d[0].size() + padding_num;
//             input_cols = input3d[0][0].size() + padding_num;
            
//             vector<vector<vector<double>>> padded_input(input_depth, vector<vector<double>>(input_rows , vector<double>(input_cols, 0.0)));
//             int start_idx = int(padding_num/2);
            
//             // try to pad evenly left and right ie 1 zero on left 2 on right 1 on top 2 on bottom
//             for (int d = 0; d < input_depth; d++) {
//                 for (int i = 0; i < input3d[0].size(); i++) {
//                     for (int j = 0; j < input3d[0][0].size() ; j++) {
//                         padded_input[d][i+start_idx][j+start_idx] = input3d[d][i][j];
//                     }
//                 }
//             }
//             _input = padded_input;
//         }

//         int output_rows = int((input_rows - filter_len) / stride) + 1;
//         int output_cols = int((input_cols - filter_len) / stride) + 1;

//         vector<vector<vector<double>>> output(num_filters, vector<vector<double>>(output_rows, vector<double>(output_cols, 0)));

//         for (int filter_idx = 0; filter_idx < filters.size(); filter_idx++){
//             auto filter = filters[filter_idx];
//             for (int i = 0; i < output_rows; i++){
//                 for (int j = 0; j < output_cols; j++){
//                     int r = i * stride;
//                     int c = j * stride;
//                     for (int k = 0; k < filter_len; k++) {
//                         for (int l = 0; l < filter_len; l++) {
//                             output[filter_idx][i][j] += _input[filter_idx][r + k][c + l] * filter[k][l];
//                         }
//                     }
//                 }
//             }
//         }

//         for (int filter_idx = 0; filter_idx < filters.size(); filter_idx++){
//             auto filter = filters[filter_idx];
//             for (int i = 0; i < output_rows; i++){
//                 for (int j = 0; j < output_cols; j++){
//                     int r = i * stride;
//                     int c = j * stride;
//                     for (int d = 0; d < input_depth; d++) {
//                         for (int k = 0; k < filter_len; k++) {
//                             for (int l = 0; l < filter_len; l++) {
//                                 output[filter_idx][i][j] += _input[d][r + k][c + l] * filter[d][k][l];
//                             }
//                         }
//                     }
//                 }
//             }
//         }

//         relu(&output);
//         _output = output;
//         return output;
//     }
    
//     vector<vector<vector<double>>> backwards(vector<vector<vector<double>>> dLdZ, double learning_rate) {
//             // dLdZ_lst has 48 8x8 dL/dZ => 24x6x6
//             // for each layer in the 48 we calculate the sum of the cube section times the single piece
//             // dLdZ is the same size as the output

//             vector<vector<vector<double>>> dLdZ_next(_input.size(), vector<vector<double>>(_input[0].size()-padding*(filter_len-1), vector<double>(_input[0][0].size()-padding*(filter_len-1))));
//             vector<vector<vector<vector<double>>>> dLdW(num_filters, vector<vector<vector<double>>>(input_depth, vector<vector<double>>(filter_len, vector<double>(filter_len, 0.0))));

//             for (int dLdZ_idx = 0; dLdZ_idx < dLdZ.size(); dLdZ_idx++) {
//                 // flatten and apply relu
//                 vector<double> dLdZ_flat;
//                 for (int i = 0; i < dLdZ[dLdZ_idx].size(); i++) {
//                     for (int j = 0; j < dLdZ[dLdZ_idx][i].size(); j++) {
//                         dLdZ_flat.push_back(max(dLdZ[dLdZ_idx][i][j],0.0)); // leakky relu
//                         // dLdZ_flat.push_back(max(dLdZ[dLdZ_idx][i][j],0.0)); // relu
//                         // dLdZ_flat.push_back(dLdZ[dLdZ_idx][i][j]); // no relu
//                     }
//                 }
//                 // 1 layer of dLdZ multiplied with 3d shapes to make 11 filter
//                 // every value in the 3d filter v will be updated such that
//                 // v = dLdZ11*a1 + dLdZ12*a3 ...
//                 vector<vector<vector<double>>> filter = filters[dLdZ_idx];
//                 int input_depth = _input.size();
//                 int input_rows = _input[0].size()+padding*(filter_len-1);
//                 int input_cols = _input[0][0].size()+padding*(filter_len-1);
//                 int filter_len = this->filter_len;
//                 int output_rows = int((input_rows - this->filter_len) / stride) + 1;
//                 int output_cols = int((input_cols - this->filter_len) / stride) + 1;
//                 int dldz_pos = 0;
//                 for (int i = 0; i < input_rows - filter_len + 1; i += stride) {
//                     for (int j = 0; j < input_cols - filter_len + 1; j += stride) {
//                         // Ssum overlap
//                         for (int d = 0; d < input_depth; d++) {
//                             for (int k = 0; k < filter_len; k++) {
//                                 for (int l = 0; l < filter_len; l++) {
//                                     dLdW[dLdZ_idx][d][k][l] += (_input[d][i + k][j + l] * dLdZ_flat[dldz_pos]); // negative because everything was getting inverted
//                                     dLdZ_next[d][i+k][j+l] += filter[d][k][l] * dLdZ_flat[dldz_pos];
//                                 }
//                             }
//                         }
//                         dldz_pos++;
//                     }
//                 }
//             }
//             // Apply dLdW to feature maps
//         for(int filter_num=0; filter_num < num_filters; filter_num++){
//             for(int d=0; d < _input.size(); d++){
//                 for(int r=0; r<filter_len; r++){
//                     for(int c=0; c<filter_len; c++){
//                         filters[filter_num][d][r][c] -= learning_rate * dLdW[filter_num][d][r][c];
//                     }
//                 }
//             }
//         }
//         return dLdZ_next;
//     }
// };


class ConvolutionLayer : public Layer
// https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9
// https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/ 
{
public:
    int filter_len;
    int stride;
    int num_filters;
    bool padding;
    vector<vector<vector<vector<double>>>> filters;
    // Tensor<double>* _input = new Tensor<double>();
    // Tensor<double>* _output = new Tensor<double>();
    Tensor<double> _input;
    Tensor<double> _output;
    int input_depth;
    AdamConv adam;

    // vector<vector<vector<double>>> bias; // bias is added to 3d result after apply every filter

    vector<double> bias; // dB is calcaulted by averaging dLdZ

    ConvolutionLayer(int num_filters, int input_depth, int filter_len, double learning_rate=0.001, int stride = 1, bool padding=0) :
            adam(num_filters, input_depth, filter_len, learning_rate)
     {
        this->num_filters = num_filters;
        this->filter_len = filter_len;
        this->stride = stride;
        this->padding = padding;

        // filters is shape (feature_map_num x input_3d_depth x filter_height x filter_width)
        this->filters.resize(num_filters, vector<vector<vector<double>>>(input_depth, vector<vector<double>>(filter_len, vector<double>(filter_len, 0.0))));
        // this->bias.resize(num_filters, vector<vector<double>>(int((input_rows - filter_len) / stride) + 1, vector<double>(int((input_cols - filter_len) / stride) + 1, 0.0)));
        this->bias.resize(num_filters, 0.0);
        // vector<vector<vector<vector<double>>>> filters(num_filters, vector<vector<vector<double>>>(input_depth, vector<vector<double>>(filter_len, vector<double>(filter_len, []() { return (double)rand() / RAND_MAX; }))));
        he_weight_init(&(this->filters), input_depth*filter_len * filter_len);
        // fill_with_random(&(this->bias), num_filters*int(((input_rows - filter_len) / stride) + 1) * (int((input_rows - filter_len) / stride) + 1));
    }

    Tensor<double> forward(Tensor<double> input3d) {
        //https://stackoverflow.com/questions/59887786/algorithim-of-how-conv2d-is-implemented-in-pytorch 
        // Get the input volume dimensions
        input_depth = input3d.depth;
        assert(input_depth > 0);

        int input_rows = input3d.rows;
        int input_cols = input3d.cols;

        if(padding){
            int padding_num = (filter_len - 1);
            input_rows += padding_num;
            input_cols += padding_num;

            Tensor<double> padded_input(input_depth, input_rows , input_cols );
            int start_idx = int(padding_num/2);
            
            // try to pad evenly left and right ie 1 zero on left 2 on right 1 on top 2 on bottom
            for (int d = 0; d < input_depth; d++) {
                for (int i = 0; i < input_rows-padding_num; i++) {
                    for (int j = 0; j < input_cols-padding_num; j++) {
                        padded_input(d, i+start_idx, j+start_idx) = input3d(d, i, j);
                    }
                }
            }
            _input = padded_input;
        } else {
            _input = input3d;
        }

        int output_rows = int((input_rows - filter_len) / stride) + 1;
        int output_cols = int((input_cols - filter_len) / stride) + 1;
        Tensor<double> output(num_filters,output_rows,output_cols);
        // output = np.zeros((self.num_filters, output_rows, output_cols))
        for (int filter_idx = 0; filter_idx < filters.size(); filter_idx++){
            auto filter = filters[filter_idx];
            for (int i = 0; i < output_rows; i++){
                for (int j = 0; j < output_cols; j++){
                    int r = i * stride;
                    int c = j * stride;
                    for (int d = 0; d < input_depth; d++) {
                        for (int k = 0; k < filter_len; k++) {
                            for (int l = 0; l < filter_len; l++) {
                                output(filter_idx, i, j) += _input(d, r + k, c + l) * filter[d][k][l];
                            }
                        }
                    }
                }
            }
        }
        // apply bias
        for(int i=0; i<num_filters; i++){
            for(int j=0; j<output_rows; j++){
                for(int k=0; k<output_cols; k++){
                    // output[i][j][k] += bias[i][j][k];
                    output(i, j, k) += bias[i];
                }
            }
        }


        relu(&output);
        _output = output;
        return output;
    }
    
    Tensor<double> backwards(Tensor<double> dLdZ) {
        assert(dLdZ.depth > 0);
                // print_tensor(dLdZ);

            // dLdZ_lst has 48 8x8 dL/dZ => 24x6x6
            // for each layer in the 48 we calculate the sum of the cube section times the single piece
            // dLdZ is the same size as the output

        Tensor<double> dLdZ_next(_input.depth, _input.rows-padding*(filter_len-1), _input.cols-padding*(filter_len-1));
        vector<vector<vector<vector<double>>>> dLdW(num_filters, vector<vector<vector<double>>>(input_depth, vector<vector<double>>(filter_len, vector<double>(filter_len, 0.0))));
        
        for (int dLdZ_idx = 0; dLdZ_idx < dLdZ.depth; dLdZ_idx++) {
            // flatten and apply relu
            vector<double> dLdZ_flat;
            for (int i = 0; i < dLdZ.rows; i++) {
                for (int j = 0; j < dLdZ.cols; j++) {
                    // dLdZ_flat.push_back(max(dLdZ[dLdZ_idx][i][j],0.01*dLdZ[dLdZ_idx][i][j])); // leakky relu
                    dLdZ_flat.push_back(max(dLdZ(dLdZ_idx, i, j),0.0)); // relu
                    // dLdZ_flat.push_back(dLdZ[dLdZ_idx][i][j]); // no relu
                }
            }  
            // 1 layer of dLdZ multiplied with 3d shapes to make 11 filter
            // every value in the 3d filter v will be updated such that
            // v = dLdZ11*a1 + dLdZ12*a3 ...
            vector<vector<vector<double>>> filter = filters[dLdZ_idx];
            int input_depth = _input.depth;
            int input_rows = _input.rows+padding*(filter_len-1);
            int input_cols = _input.cols+padding*(filter_len-1);
            int filter_len = this->filter_len;
            int output_rows = int((input_rows - this->filter_len) / stride) + 1;
            int output_cols = int((input_cols - this->filter_len) / stride) + 1;
            int dldz_pos = 0;
            for (int i = 0; i < input_rows - filter_len + 1; i += stride) {
                for (int j = 0; j < input_cols - filter_len + 1; j += stride) {
                    // Ssum overlap
                    for (int d = 0; d < input_depth; d++) {
                        for (int k = 0; k < filter_len; k++) {
                            for (int l = 0; l < filter_len; l++) {
                                dLdW[dLdZ_idx][d][k][l] += (_input(d, i + k, j + l) * dLdZ_flat[dldz_pos]); // negative because everything was getting inverted
                                dLdZ_next(d, i+k, j+l) += filter[d][k][l] * dLdZ_flat[dldz_pos];
                            }
                        }
                    }
                    dldz_pos++;
                }
            }
        }
        vector<double> dLdB(bias.size(), 0.0);

        for(int i=0; i<num_filters; i++){
            for(int j=0; j< dLdZ.rows; j++){
                for(int k=0; k < dLdZ.depth; k++){
                    // bias[i][j][k] -= learning_rate * dLdZ[i][j][k];
                    dLdB[i] +=  dLdZ(i, j, k);
                }
            }
            dLdB[i] /= (dLdZ.rows * dLdZ.cols); // average it
        }

        // adam.update(&filters, &bias, dLdW, dLdB);
        
            // Apply dLdW to feature maps
            // print_vector(dLdW[1]);
            // print_vector(filters[1]);

        for(int filter_num=0; filter_num < num_filters; filter_num++){
            for(int d=0; d < input_depth; d++){
                for(int r=0; r<filter_len; r++){
                    for(int c=0; c<filter_len; c++){
                        filters[filter_num][d][r][c] -= 0.01 * dLdW[filter_num][d][r][c];
                    }
                }
            }
            bias[filter_num] -= 0.01 * dLdB[filter_num];
        }
        return dLdZ_next;
    }
};
