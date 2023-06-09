#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
using namespace std;

//     // Softmax(){

//     // }

//     vector<double> forward(vector<double> input) {
//         last_input = input;
//         double max = -100000;
//         for (int i = 0; i < input.size(); i++) {
//             max = (max > input[i]) ? max : input[i];
//         }

//         double sum = 0.0;
//         for (int i = 0; i < input.size(); i++) {
//             sum += exp(input[i]-max);
//         }

//         for (int i = 0; i < input.size(); i++) {
//             input[i] = exp(input[i]-max) / sum;
//         }
        
//         return input;
//     }

//     vector<double> backwards(vector<double> dLdZ){ // cross_entropy dLdZ = -1/p
//     // https://github.com/AlessandroSaviolo/CNN-from-Scratch/blob/master/src/layer.py
//     // https://victorzhou.com/blog/intro-to-cnns-part-2/
//         vector<double> dLdZ_exp(dLdZ.size(), 0.0);
//         vector<double> dout_dt(dLdZ.size(), 0.0); // dout_dt is dLdZ next layer
//         double sum_exp = 0.0;
//         int label_idx;

//         for(int i=0; i < last_input.size(); i++){
//             dLdZ_exp[i] = exp(last_input[i]); 
//             sum_exp += dLdZ_exp[i];       
//             if(dLdZ[i] < 0){ // it will be negative
//                 label_idx = i;
//             }
//         }
//         // i is the label index
//         for(int i=0; i < last_input.size(); i++){
//             dout_dt[i] = -dLdZ_exp[label_idx]*dLdZ_exp[i] / (sum_exp*sum_exp);
//         }
        
//         dout_dt[label_idx] = dLdZ_exp[label_idx] * (sum_exp - dLdZ_exp[label_idx]) / (sum_exp * sum_exp);

//         for(int i=0; i < last_input.size(); i++){
//             dout_dt[i] *= dLdZ[label_idx];
//         }
//         return dout_dt;
//     }

// };


// void relu(vector<vector<vector<double>>> *input_3d, bool derivative=false) {
//     if(derivative){
//         for (int i = 0; i < input_3d->size(); i++){
//             for (int j = 0; j < (*input_3d)[i].size(); j++){
//                 for (int k = 0; k < (*input_3d)[i][j].size(); k++){
//                     (*input_3d)[i][j][k] = ((*input_3d)[i][j][k] > 0) ? 1 : 0;
//                 }
//             }
//         }
//     } else {
//         for (int i = 0; i < input_3d->size(); i++){
//             for (int j = 0; j < (*input_3d)[i].size(); j++){
//                 for (int k = 0; k < (*input_3d)[i][j].size(); k++){
//                     // (*input_3d)[i][j][k] = max((*input_3d)[i][j][k],0.001*(*input_3d)[i][j][k]); // leaky relu
//                     (*input_3d)[i][j][k] = max((*input_3d)[i][j][k],0.0); // leaky relu
//                 }
//             }
//         }
//     }
// }

// void relu(vector<double> *input, bool derivative=false) {
//     if(derivative){
//         for (int i = 0; i < input->size(); i++){
//             (*input)[i] = ((*input)[i] > 0) ? 1 : 0;
//         }
//     } else {
//         for (int i = 0; i < input->size(); i++){
//             (*input)[i] = max((*input)[i], 0.001*(*input)[i]);
//         }
//     }
// }

// vector<vector<vector<double>>> reshape(vector<double> input, int depth, int rows, int cols) {
//     vector<vector<vector<double>>> output(depth, vector<vector<double>>(rows, vector<double>(cols, 0.0)));
//     int counter = 0;
//     for(int d=0; d < depth; d++){
//         for(int r=0; r < rows; r++){
//             for(int c=0; c < cols; c++){
//                 output[d][r][c] = input[counter];
//                 counter++;
//             }
//         }
//     }
//     return output;
// }


// vector<double> min_max_scale(vector<double> input) {
//     // Find the minimum and maximum values in the input vector
//     double min = *min_element(input.begin(), input.end());
//     double max = *max_element(input.begin(), input.end());

//     // Scale the input values to the range [0,1]
//     for (int i = 0; i < input.size(); i++) {
//         input[i] = (input[i] - min) / (max - min);
//     }

//     return input;
// }


/*
class Adam{
    // https://arxiv.org/pdf/1412.6980.pdf
    // https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
public:
    double m_dw, v_dw, m_db, v_db, beta1, beta2, epsilon, learning_rate;

    Adam(double learning_rate=0.01, double beta1=0.9, double beta2=0.999, double epsilon=1e-8) {
        m_dw = v_dw = m_db = v_db = 0;
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->epsilon = epsilon;
        this->learning_rate = learning_rate;
    }

    std::pair<double, double> update(int t, double w, double b, double dw, double db) {
        // dw, db are from the current minibatch
        // momentum beta 1
        // weights
        m_dw = beta1 * m_dw + (1 - beta1) * dw; // biased momentum estimate
        // biases
        m_db = beta1 * m_db + (1 - beta1) * db; 

        // rms beta 2
        // weights
        v_dw = beta2 * v_dw + (1 - beta2) * pow(dw, 2); // bias corrected momentum estimate
        // biases
        v_db = beta2 * v_db + (1 - beta2) * pow(db, 2);

        // bias correction
        double m_dw_corr = m_dw / (1 - pow(beta1, t));
        double m_db_corr = m_db / (1 - pow(beta1, t));
        double v_dw_corr = v_dw / (1 - pow(beta2, t));
        double v_db_corr = v_db / (1 - pow(beta2, t));

        // update weights and biases
        w = w - learning_rate * (m_dw_corr / (sqrt(v_dw_corr) + epsilon));
        b = b - learning_rate * (m_db_corr / (sqrt(v_db_corr) + epsilon));

        return std::make_pair(w, b);
    }
};
*/

// class AdamFCL{
// public:
//     vector<vector<double>>  m_dw, v_dw;
//     vector<double> m_db, v_db;
//     double beta1, beta2, epsilon, learning_rate, initial_b1, initial_b2;
//     int t = 0;
//     int counter=0;
//     int iterations = 100;
//     double decay_rate = 0.8;
//     double gamma_init = 0.00001;
//     // rows = dw.size() cols = dw[0].size()

//     AdamFCL(double rows, double cols, double learning_rate=0.0005, double beta1=0.9, double beta2=0.999, double epsilon=1e-8) :
//     m_dw(rows, vector<double>(cols, 0)),
//     v_dw(rows, vector<double>(cols, 0)),
//     m_db(rows, 0),
//     v_db(rows, 0)
//     {
       
//         this->beta1 = beta1;
//         this->initial_b1 = beta1;
//         this->initial_b2 = beta2;
//         this->beta2 = beta2;
//         this->epsilon = epsilon;
//         this->learning_rate = learning_rate;
//     }

//     void update( vector<vector<double>> *w, vector<double> *b, vector<vector<double>> dw, vector<double> db) { // t is current timestep
//         // dw, db are what we would usually update params with gradient descent
//         // printArray(m_dw[0], 10);
//         // printArray(dw[0], 10);
//         // counter++;
//         // if((counter % iterations) == 0){
//         //     // beta1 *= initial_b1;
//         //     // beta2 *= initial_b2;
//         // }
//         this->t++;

//         // momentum beta 1
//         // weights
//         for(int i=0; i< (*w).size(); i++){
//             for(int j=0; j< (*w)[0].size(); j++){
//                 m_dw[i][j] = beta1 * m_dw[i][j] + (1 - beta1) * dw[i][j]; // biased momentum estimate
//                 v_dw[i][j] = beta2 * v_dw[i][j] + (1 - beta2) * pow(dw[i][j], 2); // bias corrected momentum estimate
//             }
//         }
//                 // printArray(m_dw[0], 10);

//         // cout << m_dw[7][7] << endl;
//         // print_vector(m_dw);
//         // biases
//         for(int i=0; i< (*w).size(); i++){
//             m_db[i] = beta1 * m_db[i] + (1 - beta1) * db[i]; 
//             v_db[i] = beta2 * v_db[i] + (1 - beta2) * pow(db[i], 2);
//         }
        

//         // rms beta 2
//         // weights
//         // biases
//         vector<vector<double>> m_dw_corr(m_dw.size(), vector<double>(m_dw[0].size()));
//         vector<vector<double>> v_dw_corr(m_dw.size(), vector<double>(m_dw[0].size()));
//         vector<double> m_db_corr(m_dw.size());
//         vector<double> v_db_corr(m_dw.size());

//         double denom_mw = (1 - pow(beta1, t));
//         double denom_vw = (1 - pow(beta2, t));
//         // cout << denom_mw << endl;
//         for (int i = 0; i < m_dw.size(); i++) {
//             for (int j = 0; j < m_dw[i].size(); j++) {
//                 m_dw_corr[i][j] = m_dw[i][j] / denom_mw;
//                 v_dw_corr[i][j] = v_dw[i][j] / denom_vw;
//             }
//         }

//         // bias correction
//         double denom_mb = (1 - pow(beta1, t));
//         double denom_vb = (1 - pow(beta2, t));

//         for (int j = 0; j < m_dw.size(); j++) {
//             m_db_corr[j] = m_db[j] / denom_mb;
//             v_db_corr[j] = v_db[j] / denom_vb;
//         }

//         // printArray(m_dw_corr[0], 10);
//         // update weights and biases 
//         // double gamma = gamma_init * decay_rate;
//         for(int i=0; i< (*w).size(); i++){
//             for(int j=0; j< (*w)[i].size(); j++){

//                 (*w)[i][j] -= learning_rate * (m_dw_corr[i][j] / (sqrt(v_dw_corr[i][j]) + epsilon));
//             }
//         }
//         // printArray(m_db_corr, 10);
//         // learning_rate = learning_rate * sqrt(denom_vb) / denom_mb;

//         // b -= learning_rate *  dL/dZ
//         for(int i=0; i < (*b).size(); i++){
//             (*b)[i] -=learning_rate * (m_db_corr[i] / (sqrt(v_db_corr[i]) + epsilon)); 
//         }
        
//         return;
//     }
// };


// class AdamConv{
// public:
//     vector<vector<vector<vector<double>>>>  m_dw, v_dw; // one for every weight in layer
//     vector<double> m_db, v_db; // one for every bias in layer
//     double beta1, beta2, epsilon, learning_rate, initial_b1, initial_b2;
//     int t = 0;
//     int counter=0;
//     int iterations = 100;
//     double decay_rate = 0.8;
//     double gamma_init = 0.00001;
//     int filter_depth, filter_size, num_filters;

//     AdamConv(int num_filters, int filter_depth, int filter_size, double learning_rate=0.0005, double beta1=0.9, double beta2=0.999, double epsilon=1e-8) :
//     m_dw(num_filters, vector<vector<vector<double>>>(filter_depth, vector<vector<double>>(filter_size, vector<double>(filter_size, 0)))),
//     v_dw(num_filters, vector<vector<vector<double>>>(filter_depth, vector<vector<double>>(filter_size, vector<double>(filter_size, 0)))),
//     m_db(num_filters, 0),
//     v_db(num_filters, 0)
//     {
       
//         this->beta1 = beta1;
//         // this->initial_b1 = beta1;
//         // this->initial_b2 = beta2;
//         this->beta2 = beta2;
//         this->epsilon = epsilon;
//         this->learning_rate = learning_rate;
//         this->filter_depth = filter_depth;
//         this->num_filters = num_filters;
//         this->filter_size = filter_size;
//     }

//     void update( vector<vector<vector<vector<double>>>> *w, vector<double> *b, vector<vector<vector<vector<double>>>> dw, vector<double> db) { // t is current timestep
//         // dw, db are what we would usually update params with gradient descent
//         // printArray(m_dw[0], 10);
//         // printArray(dw[0], 10);
//         // counter++;
//         // if((counter % iterations) == 0){
//         //     // beta1 *= initial_b1;
//         //     // beta2 *= initial_b2;
//         // }
//         this->t++;

//         // momentum beta 1
//         // weights
//         for(int n=0; n < num_filters; n++){
//             for(int d=0; d < filter_depth; d++){
//                 for(int j=0; j < filter_size; j++){
//                     for(int k=0; k < filter_size; k++){
//                         m_dw[n][d][j][k] = beta1 * m_dw[n][d][j][k] + (1 - beta1) * dw[n][d][j][k]; // biased momentum estimate
//                         v_dw[n][d][j][k] = beta2 * v_dw[n][d][j][k] + (1 - beta2) * pow(dw[n][d][j][k], 2); // bias corrected momentum estimate     
//                     }
//                 }
//             }
//         }
//         // printArray(m_dw[0], 10);

//         // cout << m_dw[7][7] << endl;
//         // print_vector(m_dw);
//         // biases
//         for(int i=0; i< (*w).size(); i++){
//             m_db[i] = beta1 * m_db[i] + (1 - beta1) * db[i]; 
//             v_db[i] = beta2 * v_db[i] + (1 - beta2) * pow(db[i], 2);
//         }
        

//         // rms beta 2
//         // weights
//         // biases
//         vector<vector<vector<vector<double>>>> m_dw_corr(num_filters, vector<vector<vector<double>>>(filter_depth, vector<vector<double>>(filter_size, vector<double>(filter_size, 0))));
//         vector<vector<vector<vector<double>>>> v_dw_corr(num_filters, vector<vector<vector<double>>>(filter_depth, vector<vector<double>>(filter_size, vector<double>(filter_size, 0))));
//         vector<double> m_db_corr(m_dw.size());
//         vector<double> v_db_corr(m_dw.size());

//         double denom_mw = (1 - pow(beta1, t));
//         double denom_vw = (1 - pow(beta2, t));
//         // cout << denom_mw << endl;
//         for(int n=0; n < num_filters; n++){
//             for(int d=0; d < filter_depth; d++){
//                 for(int j=0; j < filter_size; j++){
//                     for(int k=0; k < filter_size; k++){
//                         m_dw_corr[n][d][j][k] = m_dw[n][d][j][k] / denom_mw; // biased momentum estimate
//                         v_dw_corr[n][d][j][k] = v_dw[n][d][j][k] / denom_vw; // bias corrected momentum estimate     
//                     }
//                 }
//             }
//         }

//         // bias correction
//         double denom_mb = (1 - pow(beta1, t));
//         double denom_vb = (1 - pow(beta2, t));

//         for (int j = 0; j < m_dw.size(); j++) {
//             m_db_corr[j] = m_db[j] / denom_mb;
//             v_db_corr[j] = v_db[j] / denom_vb;
//         }

//         // printArray(m_dw_corr[0], 10);
//         // update weights and biases 
//         // double gamma = gamma_init * decay_rate;
//         for(int n=0; n < num_filters; n++){
//             for(int d=0; d < filter_depth; d++){
//                 for(int j=0; j < filter_size; j++){
//                     for(int k=0; k < filter_size; k++){
//                         (*w)[n][d][j][k] -= learning_rate * (m_dw_corr[n][d][j][k] / (sqrt(v_dw_corr[n][d][j][k]) + epsilon));  
//                     }
//                 }
//             }
//         }
//         // printArray(m_db_corr, 10);
//         // learning_rate = learning_rate * sqrt(denom_vb) / denom_mb;

//         // b -= learning_rate *  dL/dZ
//         for(int i=0; i < (*b).size(); i++){
//             (*b)[i] -=learning_rate * (m_db_corr[i] / (sqrt(v_db_corr[i]) + epsilon)); 
//         }
        
//         return;
//     }
// };




// void set_mask(vector<vector<double>> *mask) {
//     random_device rd;
//     mt19937 gen(rd());
//     binomial_distribution<> d(1, 0.9);

//     for(int i = 0; i < mask->size(); i++) {
//         for (int j = 0; j < (*mask)[0].size(); j++) {
//             (*mask)[i][j] = d(gen);
//         }
//     }
// }

// // 4d vector input
// void print_shape(const vector<vector<vector<vector<double>>>> &vec)
// {
//     int a = vec.size();
//     int b = vec[0].size();
//     int c = vec[0][0].size();
//     int d = vec[0][0][0].size();
//     cout << "(" << a << ", " << b << ", " << c << ", " << d << ")" << endl;
// }

// // 3d vector input
// void print_shape(const vector<vector<vector<double>>> &vec)
// {
//     int a = vec.size();
//     int b = vec[0].size();
//     int c = vec[0][0].size();
//     cout << "(" << a << ", " << b << ", " << c << ")" << endl;
// }
// // 1d vector input
// void print_shape(const vector<double> &vec)
// {
//     int a = vec.size();
//     cout << "(" << a << ")" << endl;
// }

// void he_weight_init(vector<vector<vector<vector<double>>>> *filters, int size)
// {// https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
//     default_random_engine generator;
//     normal_distribution<double> distribution(0,  sqrt(2.0 / size));

//     // set the entries of the vector to a random value
//     for (int i = 0; i < filters->size(); i++)
//     {
//         for (int j = 0; j < (*filters)[0].size(); j++)
//         {
//             for (int k = 0; k < (*filters)[0][0].size(); k++)
//             {
//                 for (int l = 0; l < (*filters)[0][0][0].size(); l++)
//                 {
//                     (*filters)[i][j][k][l] = distribution(generator);

//                     // cout << distribution(generator) << endl;
//                 }
//             }
//         }
//     }
// }

// void he_weight_init(vector<vector<vector<double>>> *bias, int size)
// {// https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
//     default_random_engine generator;
//     normal_distribution<double> distribution(0,  sqrt(2.0 / size));

//     // set the entries of the vector to a random value
//     for(int i = 0; i < (*bias).size(); i++) {
//         for(int j = 0; j < (*bias)[0].size(); j++) {
//             for(int k=0; k < (*bias)[0][0].size(); k++){
//                 (*bias)[i][j][k] = distribution(generator);
//             }
//         }
//     }
// }

// void he_weight_init(vector<double> *bias, int size)
// {// https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
//     default_random_engine generator;
//     normal_distribution<double> distribution(0,  sqrt(2.0 / size));

//     // set the entries of the vector to a random value
//     for(int i = 0; i < (*bias).size(); i++) {
//         (*bias)[i] = distribution(generator);
//     }
// }

// class ConvolutionLayer 
// // https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9
// // https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/ 
// {
// public:
//     int filter_len;
//     int stride;
//     int num_filters;
//     bool padding;
//     vector<vector<vector<vector<double>>>> filters;
//     vector<vector<vector<double>>> _input;
//     vector<vector<vector<double>>> _output;
//     int input_depth;
//     AdamConv adam;

//     // vector<vector<vector<double>>> bias; // bias is added to 3d result after apply every filter
//     vector<double> bias; // dB is calcaulted by averaging dLdZ
//     ConvolutionLayer(int num_filters, int input_depth, int filter_len, int stride = 1, bool padding=0) :
//             adam(num_filters, input_depth, filter_len)
//      {
//         this->num_filters = num_filters;
//         this->filter_len = filter_len;
//         this->stride = stride;
//         this->padding = padding;


//         // filters is shape (feature_map_num x input_3d_depth x filter_height x filter_width)
//         this->filters.resize(num_filters, vector<vector<vector<double>>>(input_depth, vector<vector<double>>(filter_len, vector<double>(filter_len, 0.0))));
//         // this->bias.resize(num_filters, vector<vector<double>>(int((input_rows - filter_len) / stride) + 1, vector<double>(int((input_cols - filter_len) / stride) + 1, 0.0)));
//         this->bias.resize(num_filters, 0.0);
//         // vector<vector<vector<vector<double>>>> filters(num_filters, vector<vector<vector<double>>>(input_depth, vector<vector<double>>(filter_len, vector<double>(filter_len, []() { return (double)rand() / RAND_MAX; }))));
//         he_weight_init(&(this->filters), input_depth*filter_len * filter_len);
//         // fill_with_random(&(this->bias), num_filters*int(((input_rows - filter_len) / stride) + 1) * (int((input_rows - filter_len) / stride) + 1));
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
//         // output = np.zeros((self.num_filters, output_rows, output_cols))
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
//         // apply bias
//         for(int i=0; i<num_filters; i++){
//             for(int j=0; j<output_rows; j++){
//                 for(int k=0; k<output_cols; k++){
//                     // output[i][j][k] += bias[i][j][k];
//                     output[i][j][k] += bias[i];
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

//         vector<vector<vector<double>>> dLdZ_next(_input.size(), vector<vector<double>>(_input[0].size()-padding*(filter_len-1), vector<double>(_input[0][0].size()-padding*(filter_len-1))));
//         vector<vector<vector<vector<double>>>> dLdW(num_filters, vector<vector<vector<double>>>(input_depth, vector<vector<double>>(filter_len, vector<double>(filter_len, 0.0))));
        
//         for (int dLdZ_idx = 0; dLdZ_idx < dLdZ.size(); dLdZ_idx++) {
//             // flatten and apply relu
//             vector<double> dLdZ_flat;
//             for (int i = 0; i < dLdZ[dLdZ_idx].size(); i++) {
//                 for (int j = 0; j < dLdZ[dLdZ_idx][i].size(); j++) {
//                     // dLdZ_flat.push_back(max(dLdZ[dLdZ_idx][i][j],0.01*dLdZ[dLdZ_idx][i][j])); // leakky relu
//                     dLdZ_flat.push_back(max(dLdZ[dLdZ_idx][i][j],0.0)); // relu
//                     // dLdZ_flat.push_back(dLdZ[dLdZ_idx][i][j]); // no relu
//                 }
//             }  
//             // 1 layer of dLdZ multiplied with 3d shapes to make 11 filter
//             // every value in the 3d filter v will be updated such that
//             // v = dLdZ11*a1 + dLdZ12*a3 ...
//             vector<vector<vector<double>>> filter = filters[dLdZ_idx];
//             int input_depth = _input.size();
//             int input_rows = _input[0].size()+padding*(filter_len-1);
//             int input_cols = _input[0][0].size()+padding*(filter_len-1);
//             int filter_len = this->filter_len;
//             int output_rows = int((input_rows - this->filter_len) / stride) + 1;
//             int output_cols = int((input_cols - this->filter_len) / stride) + 1;
//             int dldz_pos = 0;
//             for (int i = 0; i < input_rows - filter_len + 1; i += stride) {
//                 for (int j = 0; j < input_cols - filter_len + 1; j += stride) {
//                     // Ssum overlap
//                     for (int d = 0; d < input_depth; d++) {
//                         for (int k = 0; k < filter_len; k++) {
//                             for (int l = 0; l < filter_len; l++) {
//                                 dLdW[dLdZ_idx][d][k][l] += (_input[d][i + k][j + l] * dLdZ_flat[dldz_pos]); // negative because everything was getting inverted
//                                 dLdZ_next[d][i+k][j+l] += filter[d][k][l] * dLdZ_flat[dldz_pos];
//                             }
//                         }
//                     }
//                     dldz_pos++;
//                 }
//             }
//         }
//         vector<double> dLdB(bias.size(), 0.0);

//         for(int i=0; i<num_filters; i++){
//             for(int j=0; j< dLdZ[0].size(); j++){
//                 for(int k=0; k < dLdZ[0][0].size(); k++){
//                     // bias[i][j][k] -= learning_rate * dLdZ[i][j][k];
//                     dLdB[i] +=  dLdZ[i][j][k];
//                 }
//             }
//             dLdB[i] /= dLdZ[0].size() * dLdZ[0][0].size(); // average it
//         }


//         adam.update(&filters, &bias, dLdW, dLdB);
        
//             // Apply dLdW to feature maps
//             // print_vector(dLdW[1]);
//             // print_vector(filters[1]);

//         // for(int filter_num=0; filter_num < num_filters; filter_num++){
//         //     for(int d=0; d < _input.size(); d++){
//         //         for(int r=0; r<filter_len; r++){
//         //             for(int c=0; c<filter_len; c++){
//         //                 filters[filter_num][d][r][c] -= learning_rate * dLdW[filter_num][d][r][c];
//         //             }
//         //         }
//         //     }
//         //     bias[filter_num] -= learning_rate * dLdB[filter_num];
//         // }
//         return dLdZ_next;
//     }
// };



// class FullyConnectedLayer {
// public:
//     int input_size;
//     int output_size;
//     vector<vector<double>> weights; // matrix of shape (output_size, input_size)
//     vector<double> bias; // vector of shape (output_size)
//     vector<double> input_matrix;
//     // AdamFCL* adam;
//     AdamFCL adam;


//     FullyConnectedLayer(int input_size, int output_size) :
//         input_size(input_size),
//         output_size(output_size),
//         weights(output_size, vector<double>(input_size)),
//         bias(output_size, 0.0),
//         adam(output_size, input_size)
//     {
//         this->input_size = input_size;
//         this->output_size = output_size;
//         // adam = new AdamFCL(output_size, input_size);

//         // Initialize weights with random values
//         random_device rd;
//         mt19937 gen(rd());
//         normal_distribution<double> dist(0.0, sqrt(2.0 / input_size));

//         for (auto& row : this->weights) {
//             generate(row.begin(), row.end(), [&](){ return dist(gen); });
//         }

//     }

//     vector<double> forward(vector<double> input_matrix, bool dropout=false) {
//         this->input_matrix = input_matrix;
//         vector<double> outputs(output_size, 0.0);


//         if(dropout){
//             vector<vector<double>> mask(output_size, vector<double>(input_size));
//             set_mask(&mask);
//             // cout << mask[0][4];

//             for (int i = 0; i < output_size; i++) {
//                 for (int j = 0; j < input_size; j++) {
//                     outputs[i] += mask[i][j] * this->weights[i][j] * input_matrix[j];
//                 }
//                 outputs[i] += this->bias[i];
//             }

//         } else {

//             for (int i = 0; i < output_size; i++) {
//                 for (int j = 0; j < input_size; j++) {
//                     outputs[i] += this->weights[i][j] * input_matrix[j];
//                 }
//                 outputs[i] += this->bias[i];
//             }

//         }
//         relu(&outputs);
//         return outputs;
//     }
    
//     vector<double> backwards(vector<double> dLdZ, float learning_rate) {
//         // dLdA == dLdZ*relu_derivative(dLdZ)==relu(dLdZ) because of how relu works a*drelu(a) == relu(a)
        
//         // relu(&dLdZ, true); 

//         // calculate next layer dLdZ
//         vector<double> next_dLdZ(input_size, 0.0);

//         for(int c=0; c < input_size; c++){
//             for(int r=0; r < dLdZ.size(); r++){
//                 next_dLdZ[c] += weights[r][c]*dLdZ[r];
//             }
//         }

//         // calculate dLdW to update weights
//         vector<vector<double>> dLdW(this->output_size, vector<double>(this->input_size));

//         for(int r=0; r < dLdZ.size(); r++){
//             for(int c=0; c < input_size; c++){
//                 dLdW[r][c] = dLdZ[r]*input_matrix[c];
//             }
//         }

//         adam.update(&weights, &bias, dLdW, dLdZ);

//         // // w -= learning_rate * dL/dW
//         // for(int i=0; i< weights.size(); i++){
//         //     for(int j=0; j< weights[i].size(); j++){
//         //         weights[i][j] -= learning_rate * dLdW[i][j];
//         //     }
//         // }
//         // // b -= learning_rate *  dL/dZ
//         // for(int i=0; i<bias.size(); i++){
//         //     bias[i] -= learning_rate* 10 * dLdZ[i]; // make learning rate for bias larger because its smaller number smaller condition more likely to convergeto 0
//         // }
//         // printArray(bias, 10);

//         return next_dLdZ;
//     }
// };

// class Pool {
// public:
//     int pool_size;
//     int stride;
//     int input_rows = -1;
//     int input_cols = -1;
//     int num_filters = -1;
    
//     Pool(int pool_size_, int stride_) {
//         pool_size = pool_size_;
//         stride = stride_;
//     }

//     vector<vector<vector<double>>> max_pool(vector<vector<vector<double>>> input3d) {
//         int input_depth = input3d.size();
//         int input_rows = input3d[0].size();
//         int input_cols = input3d[0][0].size();
//         int output_depth = input_depth;
//         this->input_rows = input_rows;
//         this->input_cols = input_cols;
//         this->num_filters = input_depth;
//         int output_rows = (input_rows - this->pool_size) / this->stride + 1;
//         int output_cols = (input_cols - this->pool_size) / this->stride + 1;

//         vector<vector<vector<double>>> output(output_depth, vector<vector<double>>(output_rows, vector<double>(output_cols)));
//         for (int d = 0; d < output_depth; d++) {
//             for (int i = 0; i < output_rows; i++) {
//                 for (int j = 0; j < output_cols; j++) {
//                     vector<vector<double>> pool_region(this->pool_size, vector<double>(this->pool_size));
//                     for (int r = 0; r < this->pool_size; r++) {
//                         for (int c = 0; c < this->pool_size; c++) {
//                             pool_region[r][c] = input3d[d][i * this->stride + r][j * this->stride + c];
//                         }
//                     }
//                     double max_val = pool_region[0][0];
//                     for (int r = 0; r < this->pool_size; r++) {
//                         for (int c = 0; c < this->pool_size; c++) {
//                             max_val = max(max_val, pool_region[r][c]);
//                         }
//                     }
//                     output[d][i][j] = max_val;
//                 }
//             }
//         }

//         return output;
//     }

//     vector<vector<vector<double>>> upsample(vector<vector<vector<double>>> input3d) {  // 24x6x6 => 24x12x12
//         vector<vector<vector<double>>> output(this->num_filters, vector<vector<double>>(this->input_rows, vector<double>(this->input_cols)));

//         for (int filter_idx = 0; filter_idx < this->num_filters; filter_idx++) {
//             vector<vector<double>> filter_matrix = input3d[filter_idx];
//             int filter_rows = filter_matrix.size();
//             int filter_cols = filter_matrix[0].size();
//             int switch_rows = static_cast<int>(ceil(static_cast<double>(this->input_rows) / filter_rows)); // input = 5x5 filter = 2x2 then swittch every ceil(5/2)=3 rows ie [1,2] => [1,1,1,2,2] for one row
//             int switch_cols = static_cast<int>(ceil(static_cast<double>(this->input_cols) / filter_cols));
//             int r = -1, c = -1;  // index's start at 0
//             // return input3d;
//             for (int i = 0; i < this->input_rows; i++) {
//                 if ((i % switch_rows) == 0) {
//                     r += 1;
//                 }
//                 c = -1;
//                 for (int j = 0; j < this->input_cols; j++) {
//                     if ((j % switch_cols) == 0) {
//                         c += 1;
//                     }
//                     output[filter_idx][i][j] = filter_matrix[r][c];
//                 }
//             }
//         }

//         return output;
//     }
// };

// class BatchNorm1D{ // input and output are same dimensions
// // https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/ 
// public:
//     double gamma;
//     double beta;
//     vector<double> inputs;
//     double epsilon;
//     double variance;

//     BatchNorm1D( double epsilon = 1e-5){
//         // Initialize gamma and beta as 1 and 0 to start
//         this->gamma = 1;
//         this->beta = 0;
//         this->epsilon = epsilon;
//         // random_device rd;
//         // mt19937 gen(rd());
//         // uniform_real_distribution<double> dis(-1.0, 1.0);

//         // for (int i = 0; i < input_shape; i++) {
//         //     gamma[i] = dis(gen);
//         //     beta[i] = dis(gen);
//         // }
//     }

//     void batch_normalize(vector<double> *input) { // epsilon helps prevent division by 0
//         int size = (*input).size();
//         this->inputs = (*input);
//         // Compute mean
//         double mean = 0;
//         for (int i = 0; i < size; i++) {
//             mean += (*input)[i];
//         }
//         mean /= size;

//         // Compute variance
//         variance = 0;
//         for (int i = 0; i < size; i++) {
//             double diff = (*input)[i] - mean;
//             variance += diff * diff;
//         }
//         variance /= size;

//         // Compute standard deviation
//         double stddev = sqrt(variance);

//         // Normalize input
//         for (int i = 0; i < size; i++) {
//             (*input)[i] = ((*input)[i] - mean) / (stddev + epsilon);
//         }

//         // rescale 
//         for (int i = 0; i < size; i++) {
//             (*input)[i] = gamma*(*input)[i] + beta;
//         }

//     }

//     vector<double> backwards(vector<double> dLdZ, double learning_rate){
//         // https://deepnotes.io/batchnorm#backward-propagation
//         vector<double> dLdA(dLdZ.size(), 0.0); //= dLdZ * gamma / sqrt(variance+epsilon); // epsilon added to avoid division by zero
//         double dLdG = 0;
//         double dLdB = 0;

//         for(int i=0; i<dLdZ.size(); i++){
//             dLdA[i] = dLdZ[i] * gamma / sqrt(variance+epsilon);
//         }

//         for(int i=0; i<dLdZ.size(); i++){
//             dLdG += dLdZ[i] * inputs[i];
//         }
//         // dLdB == sum(dLdZ)
//         for(int i=0; i<dLdZ.size(); i++){
//             dLdB += dLdZ[i];
//         }

//         beta -= learning_rate * dLdB;
//         gamma -= learning_rate * dLdG;

//         return dLdA;
//     }
// };

// class BatchNorm3D{ // input and output are same dimensions
// public:
//     vector<double> gamma;
//     vector<double> beta;
//     vector<vector<vector<double>>> inputs;
//     double epsilon;
//     vector<double> variance;
//     int num_channels;

//     BatchNorm3D(int num_channels, double epsilon = 1e-5){
//         // 1 gamma and 1 beta per channel
//         // Initialize gamma and beta as 1 and 0 to start
//         this->gamma = vector<double>(num_channels, 1);
//         this->beta = vector<double>(num_channels, 0);
//         this->variance = vector<double>(num_channels, 0);
//         this->num_channels = num_channels;
//         this->epsilon = epsilon;
//     }

//     void batch_normalize(vector<vector<vector<double>>> *input) { // epsilon helps prevent division by 0
//         int channel_rows = (*input)[0].size();
//         int channel_cols = (*input)[0][0].size();

//         int total = channel_cols * channel_rows;

//         for(int idx=0; idx < num_channels; idx++){
        
//             this->inputs = (*input);
//             // Compute mean
//             double mean = 0;
//             for (int i = 0; i < channel_rows; i++) {
//                 for(int j=0; j<channel_cols; j++){
//                     mean += (*input)[idx][i][j];
//                 }
//             }

//             mean /= total;

//             // Compute variance
//             variance[idx] = 0;
//             for (int i = 0; i < channel_rows; i++) {
//                 for(int j=0; j<channel_cols; j++){
//                     double diff = (*input)[idx][i][j] - mean;
//                     variance[idx] += diff * diff;
//                 }
//             }

//             variance[idx] /=  total;

//             // Compute standard deviation
//             double stddev = sqrt(variance[idx]);

//             // Normalize input
//             for (int i = 0; i < channel_rows; i++) {
//                 for(int j=0; j<channel_cols; j++){
//                     (*input)[idx][i][j] = ((*input)[idx][i][j]-mean) / (stddev + epsilon);
//                 }
//             }

//             for (int i = 0; i < channel_rows; i++) {
//                 for(int j=0; j<channel_cols; j++){
//                     (*input)[idx][i][j] = gamma[idx]*(*input)[idx][i][j] + beta[idx];
//                 }
//             }

//         }

//     }

//     void backwards(vector<vector<vector<double>>> *dLdZ, double learning_rate){
//         // https://deepnotes.io/batchnorm#backward-propagation
//         // vector<vector<vector<double>>> dLdA(dLdZ.size(), vector<vector<double>>(dLdZ[0].size(), vector<double>(dLdZ[0][0].size(), 0.0)));
//         vector<double> dLdG((*dLdZ).size(), 0.0);
//         vector<double> dLdB((*dLdZ).size(), 0.0);
       
//         for(int idx=0; idx < num_channels; idx++){
//             for(int i=0; i<(*dLdZ)[0].size(); i++){
//                 for(int j=0; j<(*dLdZ)[0][0].size(); j++){
//                     dLdG[idx] += (*dLdZ)[idx][i][j] * inputs[idx][i][j];
//                 }
//             }
//             for(int i=0; i<(*dLdZ)[0].size(); i++){
//                 for(int j=0; j<(*dLdZ)[0][0].size(); j++){
//                     dLdB[idx] += (*dLdZ)[idx][i][j];
//                 }
//             }

//             for(int i=0; i<(*dLdZ)[0].size(); i++){
//                 for(int j=0; j<(*dLdZ)[0][0].size(); j++){
//                     (*dLdZ)[idx][i][j] = (*dLdZ)[idx][i][j] * gamma[idx] / sqrt(variance[idx]+epsilon);
//                 }
//             }
//         }

//         for(int i=0; i<(*dLdZ).size(); i++) {
//             beta[i] -= learning_rate * dLdB[i];
//             gamma[i] -= learning_rate * dLdG[i];
//         }
//     }
// };

// void dropout3d(double probability, vector<vector<vector<double>>>& inputVector)
// {
//     // Initialize random number generator
//     random_device rd;
//     mt19937 gen(rd());
//     uniform_real_distribution<> dis(0, 1);

//     // Iterate over each element in the 3D vector and zero out with given probability
//     for (auto& i : inputVector) {
//         for (auto& j : i) {
//             for (auto& k : j) {
//                 double randomNum = dis(gen);
//                 if (randomNum < probability) {
//                     k = 0;
//                 }
//             }
//         }
//     }
// }
