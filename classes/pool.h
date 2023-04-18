
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
using namespace std;

class Pool {
public:
    int pool_size;
    int stride;
    int input_rows = -1;
    int input_cols = -1;
    int num_filters = -1;
    
    Pool(int pool_size_, int stride_) {
        pool_size = pool_size_;
        stride = stride_;
    }

    vector<vector<vector<double>>> max_pool(vector<vector<vector<double>>> input3d) {
        int input_depth = input3d.size();
        int input_rows = input3d[0].size();
        int input_cols = input3d[0][0].size();
        int output_depth = input_depth;
        this->input_rows = input_rows;
        this->input_cols = input_cols;
        this->num_filters = input_depth;
        int output_rows = (input_rows - this->pool_size) / this->stride + 1;
        int output_cols = (input_cols - this->pool_size) / this->stride + 1;

        vector<vector<vector<double>>> output(output_depth, vector<vector<double>>(output_rows, vector<double>(output_cols)));
        for (int d = 0; d < output_depth; d++) {
            for (int i = 0; i < output_rows; i++) {
                for (int j = 0; j < output_cols; j++) {
                    vector<vector<double>> pool_region(this->pool_size, vector<double>(this->pool_size));
                    for (int r = 0; r < this->pool_size; r++) {
                        for (int c = 0; c < this->pool_size; c++) {
                            pool_region[r][c] = input3d[d][i * this->stride + r][j * this->stride + c];
                        }
                    }
                    double max_val = pool_region[0][0];
                    for (int r = 0; r < this->pool_size; r++) {
                        for (int c = 0; c < this->pool_size; c++) {
                            max_val = max(max_val, pool_region[r][c]);
                        }
                    }
                    output[d][i][j] = max_val;
                }
            }
        }

        return output;
    }

    vector<vector<vector<double>>> upsample(vector<vector<vector<double>>> input3d) {  // 24x6x6 => 24x12x12
        vector<vector<vector<double>>> output(this->num_filters, vector<vector<double>>(this->input_rows, vector<double>(this->input_cols)));

        for (int filter_idx = 0; filter_idx < this->num_filters; filter_idx++) {
            vector<vector<double>> filter_matrix = input3d[filter_idx];
            int filter_rows = filter_matrix.size();
            int filter_cols = filter_matrix[0].size();
            int switch_rows = static_cast<int>(ceil(static_cast<double>(this->input_rows) / filter_rows)); // input = 5x5 filter = 2x2 then swittch every ceil(5/2)=3 rows ie [1,2] => [1,1,1,2,2] for one row
            int switch_cols = static_cast<int>(ceil(static_cast<double>(this->input_cols) / filter_cols));
            int r = -1, c = -1;  // index's start at 0
            // return input3d;
            for (int i = 0; i < this->input_rows; i++) {
                if ((i % switch_rows) == 0) {
                    r += 1;
                }
                c = -1;
                for (int j = 0; j < this->input_cols; j++) {
                    if ((j % switch_cols) == 0) {
                        c += 1;
                    }
                    output[filter_idx][i][j] = filter_matrix[r][c];
                }
            }
        }

        return output;
    }
};